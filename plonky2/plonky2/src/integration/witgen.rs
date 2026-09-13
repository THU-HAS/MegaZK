//! GPU witness scheduling, buffers, public-input index preparation, and graph replay.
//!
//! This module owns witness-specific metadata and device buffers. It does not initialize quotient
//! state, mutate transcript state, or evaluate proof constraints.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::io::{self, Write};
use std::mem;
use std::ptr;
use std::time::Instant;

use boojum::field::goldilocks::{GoldilocksField as GoldilocksFieldBoojum, GoldilocksField};
use boojum_cuda::generate::NEW_FUNC_MAP;
use cudart::device::device_synchronize;
use cudart::execution::{CudaLaunchConfig, KernelFunction};
use cudart::graph::{CudaGraph, CudaGraphExec, CudaGraphNode, CudaKernelNodeParams};
use cudart::memory::{memory_copy_async, memory_copy_async_w_offset, memory_get_info, DeviceAllocation};
use cudart::stream::CudaStream;

use super::overlay::{self, DeviceView, ScratchCursor};
use super::{PolyLayout, PolySegment};
use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::iop::generator::WitnessGeneratorRef;
use crate::iop::target::Target;

cudart::cuda_kernel_signature_arguments_and_function!(
    WitgenGen,
    witness: *mut GoldilocksFieldBoojum,
    rep_map: *const u32,
    params: *const usize,
    write_start: usize,
    read_start: usize,
    param_start: usize,
    length: usize,
);

cudart::cuda_kernel_signature_arguments_and_function!(
    WitgenScatter,
    full_witness: *mut GoldilocksFieldBoojum,
    witness: *const GoldilocksFieldBoojum,
    representative_map: *const u32,
);

#[derive(Copy, Clone)]
struct StreamedKernel {
    func: WitgenGenSignature,
    grid_x: u32,
    block_x: u32,
    write_start: usize,
    read_start: usize,
    params_start: usize,
    len: usize,
}

struct StreamedWitgen {
    witness: *mut GoldilocksFieldBoojum,
    read_map: *const u32,
    params: *const usize,
    kernels: Vec<StreamedKernel>,
}

/// Scatter is a regular launch, not a graph node, so `representative_map` is not
/// captured and can be freed after the last replay.
struct ScatterKernel {
    full: *mut GoldilocksFieldBoojum,
    witness: *const GoldilocksFieldBoojum,
    rep: *const u32,
    grid: u32,
    block: u32,
}

fn empty_device_alloc<T>() -> DeviceAllocation<T> {
    unsafe { DeviceAllocation::from_raw_parts(ptr::null_mut(), 0) }
}

#[derive(Copy, Clone)]
struct WitnessKernelBatch {
    write_start: u32,
    read_start: u32,
    params_start: u32,
    len: u32,
}

pub(crate) struct WitnessPlan {
    witness_input_count: usize,
    generator_count: usize,
    public_input_representatives: Vec<usize>,
    layers: Vec<BTreeMap<String, WitnessKernelBatch>>,
    write_map: Vec<u32>,
    read_map: Vec<u32>,
    graph_execs: Vec<CudaGraphExec>,
    streamed: Option<StreamedWitgen>,
    scatter: Option<ScatterKernel>,
}

impl WitnessPlan {
    pub(crate) fn new() -> Self {
        Self {
            witness_input_count: 0,
            generator_count: 0,
            public_input_representatives: Vec::new(),
            layers: Vec::new(),
            write_map: Vec::new(),
            read_map: Vec::new(),
            graph_execs: Vec::new(),
            streamed: None,
            scatter: None,
        }
    }

    #[inline]
    pub(crate) fn has_live_graph(&self) -> bool {
        !self.graph_execs.is_empty()
    }

    #[inline]
    pub(crate) fn witness_input_count(&self) -> usize {
        self.witness_input_count
    }

    #[inline]
    pub(crate) fn generator_count(&self) -> usize {
        self.generator_count
    }

    #[inline]
    pub(crate) fn record_public_input_representative(&mut self, representative: usize) {
        self.public_input_representatives.push(representative);
    }
}

pub(crate) struct WitnessBuffers {
    pub(crate) public_input_indices: DeviceAllocation<usize>,
    pub(crate) witness: DeviceView<GoldilocksFieldBoojum>,
    /// Witgen-only scatter map. Not captured by the generator graph; dropped
    /// after replay so extra FRI can sit beside live-graph SoA.
    pub(crate) representative_map: DeviceAllocation<u32>,
    pub(crate) read_map: DeviceView<u32>,
    pub(crate) params: DeviceView<usize>,
}

impl WitnessBuffers {
    pub(crate) fn new(num_public_inputs: usize) -> Self {
        Self {
            public_input_indices: DeviceAllocation::<usize>::alloc(num_public_inputs).unwrap(),
            witness: DeviceView::empty(),
            representative_map: empty_device_alloc(),
            read_map: DeviceView::empty(),
            params: DeviceView::empty(),
        }
    }

    pub(crate) fn unbind_device_views(&mut self) {
        self.witness = DeviceView::empty();
        self.read_map = DeviceView::empty();
        self.params = DeviceView::empty();
    }
}

pub(crate) struct WitnessService;

impl WitnessService {
    pub(crate) fn prepare_public_input_indices(
        plan: &mut WitnessPlan,
        buffers: &mut WitnessBuffers,
        public_inputs: &[Target],
        representative_map: &[usize],
        num_wires: usize,
        degree: usize,
        stream: &CudaStream,
    ) -> Vec<usize> {
        let public_inputs_index = public_inputs
            .iter()
            .map(|target| {
                let index = target.index(num_wires, degree);
                let rep_index = representative_map[index];
                plan.record_public_input_representative(rep_index);
                if rep_index < num_wires * degree {
                    (rep_index / num_wires) + (rep_index % num_wires) * degree
                } else {
                    rep_index
                }
            })
            .collect::<Vec<_>>();
        memory_copy_async(
            &mut buffers.public_input_indices,
            &public_inputs_index,
            stream,
        )
        .unwrap();
        public_inputs_index
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build<F: RichField + Extendable<D>, const D: usize>(
        plan: &mut WitnessPlan,
        buffers: &mut WitnessBuffers,
        layout: &PolyLayout,
        fp_inputs: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        scratch: &mut DeviceAllocation<u8>,
        input_targets: &[Target],
        generators: &[WitnessGeneratorRef<F, D>],
        generator_indices_by_watches: &BTreeMap<usize, Vec<usize>>,
        representative_map: &Vec<usize>,
        num_wires: usize,
        degree: usize,
        _degree_bits: usize,
        post_witgen_bytes: usize,
    ) {
        let s = Instant::now();
        let n = generators.len();
        plan.generator_count = n;
        println!("Witness generator count: {}", plan.generator_count());
        println!("num_wires: {}", num_wires);

        let mut remaining_generators = n;
        let mut pred: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        let mut succ: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        let mut target_producer: BTreeMap<usize, usize> = BTreeMap::new();
        let mut cur_generators = Vec::new();
        let mut next_generators = Vec::new();
        let mut indeg_target = vec![0; generators.len()];

        for i in 0..generators.len() {
            let unique: HashSet<usize> = generators[i]
                .0
                .watch_list()
                .iter()
                .map(|target| representative_map[target.index(num_wires, degree)])
                .collect();
            indeg_target[i] = unique.len();
            if indeg_target[i] == 0 {
                cur_generators.push(i);
            }
        }

        for t in input_targets {
            let rep = representative_map[t.index(num_wires, degree)];
            if target_producer.contains_key(&rep) {
                continue;
            }
            target_producer.insert(rep, usize::MAX);

            if let Some(watchers) = generator_indices_by_watches.get(&rep) {
                for &watching_generator_idx in watchers {
                    indeg_target[watching_generator_idx] -= 1;
                    if indeg_target[watching_generator_idx] == 0 {
                        cur_generators.push(watching_generator_idx);
                    }
                }
            }
        }

        while !cur_generators.is_empty() {
            next_generators.clear();
            for &generator_idx in &cur_generators {
                for t in &generators[generator_idx].0.output_targets() {
                    let rep = representative_map[t.index(num_wires, degree)];
                    if target_producer.contains_key(&rep) {
                        continue;
                    }
                    target_producer.insert(rep, generator_idx);
                    if let Some(watchers) = generator_indices_by_watches.get(&rep) {
                        for &watching_generator_idx in watchers {
                            indeg_target[watching_generator_idx] -= 1;
                            if indeg_target[watching_generator_idx] == 0 {
                                next_generators.push(watching_generator_idx);
                            }
                        }
                    }
                }
            }
            std::mem::swap(&mut cur_generators, &mut next_generators);
        }

        for (i, generator) in generators.iter().enumerate() {
            for target in generator.0.watch_list() {
                let rep = representative_map[target.index(num_wires, degree)];
                if let Some(&producer) = target_producer.get(&rep) {
                    if producer != usize::MAX && producer != i {
                        pred[i].insert(producer);
                        succ[producer].insert(i);
                    }
                }
            }
        }

        let mut asap = vec![0usize; n];
        let mut indeg: Vec<usize> = pred.iter().map(|p| p.len()).collect();
        let mut queue = VecDeque::new();
        for i in 0..n {
            if indeg[i] == 0 {
                queue.push_back(i);
                remaining_generators -= 1;
            }
        }
        while let Some(u) = queue.pop_front() {
            for &v in &succ[u] {
                asap[v] = asap[v].max(asap[u] + 1);
                indeg[v] -= 1;
                if indeg[v] == 0 {
                    queue.push_back(v);
                    remaining_generators -= 1;
                }
            }
        }
        assert_eq!(
            remaining_generators, 0,
            "{} generators weren't run",
            remaining_generators
        );

        let max_level = *asap.iter().max().unwrap();
        remaining_generators = n;
        let mut alap = vec![max_level; n];
        let mut outdeg: Vec<usize> = succ.iter().map(|s| s.len()).collect();
        let mut queue = VecDeque::new();
        for i in 0..n {
            if outdeg[i] == 0 {
                queue.push_back(i);
                remaining_generators -= 1;
            }
        }
        while let Some(u) = queue.pop_front() {
            for &p in &pred[u] {
                alap[p] = alap[p].min(alap[u] - 1);
                outdeg[p] -= 1;
                if outdeg[p] == 0 {
                    queue.push_back(p);
                    remaining_generators -= 1;
                }
            }
        }
        assert_eq!(
            remaining_generators, 0,
            "{} generators weren't run",
            remaining_generators
        );

        let mut layer: Vec<BTreeMap<String, Vec<usize>>> = vec![BTreeMap::new(); max_level + 1];
        // Default stays "custom" so SHA/ECDSA keep fat GPU batches. Opt-in ASAP:
        // PLONKY2_WITGEN_SCHEDULING=asap
        let scheduling = std::env::var("PLONKY2_WITGEN_SCHEDULING")
            .unwrap_or_else(|_| String::from("custom"));
        let mut chain_fast = false;
        if scheduling == "asap" {
            println!("Scheduling: asap");
            for g in 0..n {
                layer[asap[g]]
                    .entry(generators[g].0.id())
                    .or_default()
                    .push(g);
            }
        } else if scheduling == "alap" {
            println!("Scheduling: alap");
            for g in 0..n {
                layer[alap[g]]
                    .entry(generators[g].0.id())
                    .or_default()
                    .push(g);
            }
        } else {
            let mut scheduled: Vec<usize> = vec![usize::MAX; n];
            let slack: Vec<usize> = (0..n).map(|i| alap[i] - asap[i]).collect();

            for g in 0..n {
                if slack[g] == 0 {
                    let level = asap[g];
                    let generator_type = generators[g].0.id();
                    layer[level].entry(generator_type).or_default().push(g);
                    scheduled[g] = level;
                }
            }

            let remaining_after_critical = scheduled
                .iter()
                .filter(|&&level| level == usize::MAX)
                .count();
            // slack=0-empty is the original predicate, but Fact/Fib still have
            // millions of ConstantGenerators with slack>0. Those circuits are the
            // deep ones (max_level ≈ num, >> SHA's ~2885). Place leftovers at ASAP
            // (legal) and skip the custom fill; SHA stays below the level cutoff.
            const CHAIN_FAST_MIN_LEVELS: usize = 8000;
            println!("ASAP max_level: {}", max_level);
            if remaining_after_critical == 0 || max_level > CHAIN_FAST_MIN_LEVELS {
                if remaining_after_critical != 0 {
                    let mut remaining_types: BTreeMap<String, usize> = BTreeMap::new();
                    for g in 0..n {
                        if scheduled[g] == usize::MAX {
                            *remaining_types
                                .entry(generators[g].0.id())
                                .or_default() += 1;
                            layer[asap[g]]
                                .entry(generators[g].0.id())
                                .or_default()
                                .push(g);
                            scheduled[g] = asap[g];
                        }
                    }
                    println!(
                        "chain-fast remaining-at-asap: {} / {}",
                        remaining_after_critical, n
                    );
                    for (ty, count) in remaining_types {
                        println!("  remaining type {}: {}", ty, count);
                    }
                }
                chain_fast = true;
                println!("Scheduling: custom-chain-fast");
                let _ = io::stdout().flush();
            } else {
                println!("Scheduling: custom");
                println!(
                    "slack-0 remaining: {} / {}",
                    remaining_after_critical, n
                );
                let _ = io::stdout().flush();
                // Restore the original collect-then-schedule scan. Typed-scan mutated
                // `scheduled` while walking a type-pool, so later generators of the same
                // type saw preds already placed on this level and were pushed to leftover.
                // MVM 2000² / 4000² then failed FRI (`evals[x_index] == old_eval`).
                // Index remaining by type so we still skip unrelated ids (O(remaining)
                // per pinned type, not O(n)), but collect a full candidate set before
                // committing any of them — same rule as `for g in 0..n`.
                let mut remaining_by_type: BTreeMap<String, Vec<usize>> = BTreeMap::new();
                for g in 0..n {
                    if scheduled[g] == usize::MAX {
                        remaining_by_type
                            .entry(generators[g].0.id())
                            .or_default()
                            .push(g);
                    }
                }
                for (ty, ids) in &remaining_by_type {
                    println!("  remaining type {}: {}", ty, ids.len());
                }
                for level in 0..=max_level {
                    let types: Vec<String> = layer[level].keys().cloned().collect();
                    for generator_type in types {
                        let Some(pool) = remaining_by_type.get_mut(&generator_type) else {
                            continue;
                        };
                        if pool.is_empty() {
                            continue;
                        }
                        let mut candidates = Vec::new();
                        let mut still = Vec::new();
                        for &g in pool.iter() {
                            if asap[g] <= level
                                && level <= alap[g]
                                && pred[g]
                                    .iter()
                                    .all(|&p| scheduled[p] < level || alap[p] < level)
                            {
                                candidates.push(g);
                            } else {
                                still.push(g);
                            }
                        }
                        for g in candidates {
                            layer[level]
                                .entry(generator_type.clone())
                                .or_default()
                                .push(g);
                            scheduled[g] = level;
                        }
                        *pool = still;
                    }
                }

                let mut remaining: HashSet<usize> =
                    remaining_by_type.into_values().flatten().collect();
                println!("custom leftover: {}", remaining.len());
                while !remaining.is_empty() {
                    let mut level_buckets: BTreeMap<usize, BTreeMap<String, Vec<usize>>> =
                        BTreeMap::new();
                    for &g in &remaining {
                        let mut r = asap[g];
                        let l = alap[g];
                        for &p in &pred[g] {
                            if scheduled[p] != usize::MAX {
                                r = r.max(scheduled[p] + 1);
                            } else {
                                r = r.max(alap[p] + 1);
                            }
                        }
                        for level in r..=l {
                            let generator_type = generators[g].0.id();
                            level_buckets
                                .entry(level)
                                .or_default()
                                .entry(generator_type)
                                .or_default()
                                .push(g);
                        }
                    }

                    let mut best_level = None;
                    let mut best_type = None;
                    let mut best_group: Vec<usize> = Vec::new();
                    for (&level, types) in &level_buckets {
                        for (generator_type, group) in types {
                            if group.len() > best_group.len() {
                                best_level = Some(level);
                                best_type = Some(generator_type.clone());
                                best_group = group.clone();
                            }
                        }
                    }

                    let level = best_level.expect("no schedulable generators");
                    let generator_type = best_type.unwrap();
                    for g in best_group {
                        scheduled[g] = level;
                        layer[level]
                            .entry(generator_type.clone())
                            .or_default()
                            .push(g);
                        remaining.remove(&g);
                    }
                }
            }
        }

        plan.write_map = vec![u32::MAX; representative_map.len()];
        plan.read_map = vec![0; representative_map.len()];
        let mut write_starts = Vec::<u32>::new();
        let mut read_starts = Vec::<u32>::new();
        let mut param_starts = Vec::<u32>::new();
        let mut witness_params = vec![0usize; representative_map.len()];

        read_starts.push(0);
        write_starts.push(input_targets.len() as u32);
        param_starts.push(0);

        for (i, target) in input_targets.iter().enumerate() {
            let rep = representative_map[target.index(num_wires, degree)];
            plan.write_map[rep] = i as u32;
        }

        for map in layer {
            let mut generator_batches = BTreeMap::new();
            for (generator_type, generator_idxs) in map {
                let mut max_in = 0;
                let mut max_out = 0;
                let mut max_params = 0;
                let num_generators = generator_idxs.len();
                for (i, generator_idx) in generator_idxs.into_iter().enumerate() {
                    let deps = generators[generator_idx].0.watch_list();
                    let outs = generators[generator_idx].0.output_targets();
                    let params = generators[generator_idx].0.get_params(
                        representative_map,
                        num_wires,
                        degree,
                    );
                    if deps.len() > max_in {
                        max_in = deps.len();
                    }
                    if outs.len() > max_out {
                        max_out = outs.len();
                    }
                    if params.len() > max_params {
                        max_params = params.len();
                    }
                    max_in = max_in.max(deps.len());
                    max_out = max_out.max(outs.len());
                    for (j, out) in outs.into_iter().enumerate() {
                        let rep = representative_map[out.index(num_wires, degree)];
                        if plan.write_map[rep] == u32::MAX {
                            plan.write_map[rep] =
                                write_starts.last().unwrap() + (i + j * num_generators) as u32;
                        }
                    }
                    let read_start = *read_starts.last().unwrap();
                    for (j, dep) in deps.into_iter().enumerate() {
                        plan.read_map[read_start as usize + i + j * num_generators] =
                            plan.write_map[representative_map[dep.index(num_wires, degree)]];
                    }
                    let param_start = *param_starts.last().unwrap();
                    for (j, param) in params.into_iter().enumerate() {
                        witness_params[param_start as usize + i + j * num_generators] = param;
                    }
                }

                let read_start = *read_starts.last().unwrap();
                read_starts.push(read_start + (num_generators * max_in) as u32);
                let write_start = *write_starts.last().unwrap();
                write_starts.push(write_start + (num_generators * max_out) as u32);
                let params_start = *param_starts.last().unwrap();
                param_starts.push(params_start + (num_generators * max_params) as u32);
                generator_batches.insert(
                    generator_type,
                    WitnessKernelBatch {
                        write_start,
                        read_start,
                        params_start,
                        len: num_generators as u32,
                    },
                );
            }
            plan.layers.push(generator_batches);
        }

        println!("Total read size: {}", read_starts.last().unwrap());
        println!("Total write size: {}", write_starts.last().unwrap());
        println!("Total param size: {}", param_starts.last().unwrap());

        let public_input_indices = plan
            .public_input_representatives
            .iter()
            .map(|&rep_idx| plan.write_map[rep_idx] as usize)
            .collect::<Vec<_>>();

        let stream = CudaStream::default();
        memory_copy_async(
            &mut buffers.public_input_indices,
            &public_input_indices,
            &stream,
        )
        .unwrap();

        plan.read_map
            .truncate(*read_starts.last().unwrap() as usize);
        let read_len = plan.read_map.len();
        let witness_len = *write_starts.last().unwrap() as usize;
        let rep_map_len = num_wires * degree;
        witness_params.truncate(*param_starts.last().unwrap() as usize);
        let params_len = witness_params.len();
        buffers.unbind_device_views();
        let captured = overlay::witgen_captured_bytes(read_len, witness_len, params_len);
        let scatter_map_bytes = rep_map_len * mem::size_of::<u32>();
        println!(
            "Witgen captured SoA: {:.2} MiB (graph addresses); scatter map: {:.2} MiB (drop after replay); FRI post: {:.2} MiB",
            captured as f64 / (1024.0 * 1024.0),
            scatter_map_bytes as f64 / (1024.0 * 1024.0),
            post_witgen_bytes as f64 / (1024.0 * 1024.0)
        );
        overlay::ensure_scratch_bytes(scratch, captured);
        {
            let mut cursor = ScratchCursor::new(scratch);
            buffers.read_map = cursor.take(read_len);
            buffers.witness = cursor.take(witness_len);
            buffers.params = cursor.take(params_len);
        }
        let old_map = mem::replace(
            &mut buffers.representative_map,
            DeviceAllocation::<u32>::alloc(rep_map_len).unwrap(),
        );
        drop(old_map);
        memory_copy_async(&mut buffers.read_map, &plan.read_map, &stream).unwrap();

        let trans_map = (0..(num_wires * degree))
            .map(|x| plan.write_map[representative_map[x]])
            .collect::<Vec<_>>();
        memory_copy_async(&mut buffers.representative_map, &trans_map, &stream).unwrap();

        memory_copy_async(&mut buffers.params, &witness_params, &stream).unwrap();
        stream.synchronize().unwrap();

        println!("Total levels: {}", plan.layers.len());
        let _ = io::stdout().flush();

        Self::build_graph(
            plan,
            buffers,
            layout,
            fp_inputs,
            num_wires,
            degree,
            chain_fast,
            post_witgen_bytes,
        );
        // CPU maps are only needed to assemble SoA + the captured graph. Drop them
        // here so munmap is charged to sorting, not prove/cm1.
        let _ = mem::take(&mut plan.write_map);
        let _ = mem::take(&mut plan.read_map);
        let _ = mem::take(&mut plan.layers);
        println!("Sorting took: {:?}", s.elapsed());
        let _ = io::stdout().flush();
        plan.witness_input_count = input_targets.len();
    }

    fn build_graph(
        plan: &mut WitnessPlan,
        buffers: &WitnessBuffers,
        layout: &PolyLayout,
        fp_inputs: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        num_wires: usize,
        degree: usize,
        chain_fast: bool,
        post_witgen_bytes: usize,
    ) {
        let s = Instant::now();
        let n_layers = plan.layers.len();
        // SHA/ECDSA (~3k levels) keep one captured graph. Deep Fact/Fib chains
        // cannot keep hundreds of instantiated graphs resident (Fib C 128×8192
        // hit ErrorMemoryAllocation at 24GB). Stream those kernels instead.
        const STREAMED_LAYER_THRESHOLD: usize = 40_000;
        if n_layers > STREAMED_LAYER_THRESHOLD {
            Self::build_streamed(plan, buffers, layout, fp_inputs, num_wires, degree);
            println!(
                "Witness exec: streamed ({} levels, chain_fast={}) in {:?}",
                n_layers,
                chain_fast,
                s.elapsed()
            );
            return;
        }
        if chain_fast {
            println!("Witness graph: single (chain-fast)");
        } else {
            println!("Witness graph: single");
        }
        let chunk = n_layers.max(1);

        let mut start = 0usize;
        loop {
            let end = start.saturating_add(chunk).min(n_layers);
            let is_last = end == n_layers;
            let mut graph = CudaGraph::new().unwrap();
            let mut previous_nodes: Vec<CudaGraphNode> = Vec::new();

            for generator_batches in &plan.layers[start..end] {
                let mut level_nodes: Vec<CudaGraphNode> = Vec::new();
                for (generator_id, &batch) in generator_batches {
                    let function = NEW_FUNC_MAP.get(generator_id.as_str()).unwrap();
                    let thread_x: u32 = 128;
                    let block_x: u32 = (batch.len + thread_x - 1) / thread_x;
                    let witness = buffers.witness.as_ptr();
                    let rep_map = buffers.read_map.as_ptr();
                    let index_params = buffers.params.as_ptr();

                    let mut kernel = CudaKernelNodeParams::new(*function as *const c_void)
                        .grid_dim(block_x)
                        .block_dim(thread_x)
                        .arg(&witness)
                        .arg(&rep_map)
                        .arg(&index_params)
                        .arg(&batch.write_start)
                        .arg(&batch.read_start)
                        .arg(&batch.params_start)
                        .arg(&batch.len);
                    let node = graph
                        .add_kernel_node(&kernel.build(), &previous_nodes)
                        .unwrap();
                    level_nodes.push(node);
                }
                previous_nodes = level_nodes;
            }
            let _ = previous_nodes;

            plan.graph_execs.push(graph.instantiate().unwrap());
            if is_last {
                break;
            }
            start = end;
        }

        plan.scatter = Some(Self::make_scatter(buffers, layout, fp_inputs, num_wires, degree));
        println!(
            "CUDA graph built in: {:?} ({} graph(s), scatter not captured)",
            s.elapsed(),
            plan.graph_execs.len()
        );
        let (free, _) = memory_get_info().unwrap();
        let map_bytes = buffers.representative_map.len() * mem::size_of::<u32>();
        let predicted = free.saturating_add(map_bytes);
        println!(
            "Live-graph extra FRI predict: free={:.0} MiB + scatter_map={:.0} MiB = {:.0} vs post={:.0} + slack={:.0}",
            free as f64 / (1024.0 * 1024.0),
            map_bytes as f64 / (1024.0 * 1024.0),
            predicted as f64 / (1024.0 * 1024.0),
            post_witgen_bytes as f64 / (1024.0 * 1024.0),
            overlay::FRI_KEEP_SOA_SLACK as f64 / (1024.0 * 1024.0)
        );
        Self::warmup_graph_first_launch(plan);
    }

    fn make_scatter(
        buffers: &WitnessBuffers,
        layout: &PolyLayout,
        fp_inputs: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        num_wires: usize,
        degree: usize,
    ) -> ScatterKernel {
        ScatterKernel {
            full: unsafe { fp_inputs.as_mut_ptr().add(layout.fp_offset(PolySegment::Wires)) },
            witness: buffers.witness.as_ptr(),
            rep: buffers.representative_map.as_ptr(),
            grid: degree as u32,
            block: num_wires as u32,
        }
    }

    fn launch_scatter(scatter: &ScatterKernel, stream: &CudaStream) {
        extern "C" {
            fn generate_full_witness_kernel(
                full_witness: *mut GoldilocksFieldBoojum,
                witness: *const GoldilocksFieldBoojum,
                representative_map: *const u32,
            );
        }
        let config = CudaLaunchConfig::basic(scatter.grid, scatter.block, stream);
        let args = WitgenScatterArguments::new(scatter.full, scatter.witness, scatter.rep);
        WitgenScatterFunction(generate_full_witness_kernel)
            .launch(&config, &args)
            .unwrap();
    }

    /// Pay the first non-graph kernel (full scatter) during *build*, not prove.
    fn warmup_graph_first_launch(plan: &WitnessPlan) {
        if plan.graph_execs.is_empty() {
            return;
        }
        let s = Instant::now();
        let stream = CudaStream::default();
        for graph_exec in &plan.graph_execs {
            graph_exec.launch(&stream).unwrap();
        }
        stream.synchronize().unwrap();
        let replay = s.elapsed();
        if let Some(scatter) = plan.scatter.as_ref() {
            Self::launch_scatter(scatter, &stream);
            stream.synchronize().unwrap();
        }
        println!(
            "  graph build warmup: replay {:?} scatter {:?}",
            replay,
            s.elapsed().saturating_sub(replay)
        );
    }

    fn build_streamed(
        plan: &mut WitnessPlan,
        buffers: &WitnessBuffers,
        layout: &PolyLayout,
        fp_inputs: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        num_wires: usize,
        degree: usize,
    ) {
        let witness = buffers.witness.as_ptr() as *mut GoldilocksFieldBoojum;
        let read_map = buffers.read_map.as_ptr();
        let params = buffers.params.as_ptr();
        let mut kernels = Vec::with_capacity(plan.layers.len());
        for generator_batches in &plan.layers {
            for (generator_id, &batch) in generator_batches {
                let function = *NEW_FUNC_MAP.get(generator_id.as_str()).unwrap();
                let thread_x: u32 = 128;
                let block_x: u32 = (batch.len + thread_x - 1) / thread_x;
                kernels.push(StreamedKernel {
                    func: function,
                    grid_x: block_x,
                    block_x: thread_x,
                    write_start: batch.write_start as usize,
                    read_start: batch.read_start as usize,
                    params_start: batch.params_start as usize,
                    len: batch.len as usize,
                });
            }
        }
        plan.streamed = Some(StreamedWitgen {
            witness,
            read_map,
            params,
            kernels,
        });
        plan.scatter = Some(Self::make_scatter(buffers, layout, fp_inputs, num_wires, degree));
    }

    fn replay_streamed(streamed: &StreamedWitgen, stream: &CudaStream) {
        for k in &streamed.kernels {
            let config = CudaLaunchConfig::basic(k.grid_x, k.block_x, stream);
            let args = WitgenGenArguments::new(
                streamed.witness,
                streamed.read_map,
                streamed.params,
                k.write_start,
                k.read_start,
                k.params_start,
                k.len,
            );
            WitgenGenFunction(k.func).launch(&config, &args).unwrap();
        }
    }

    #[inline]
    pub(crate) fn upload_inputs<F: RichField>(buffers: &mut WitnessBuffers, inputs: &[F]) {
        let length = inputs.len();
        if length != 0 {
            let s = Instant::now();
            let inputs = unsafe { mem::transmute::<&[F], &[GoldilocksField]>(inputs) };
            let stream = CudaStream::default();
            memory_copy_async_w_offset(&mut buffers.witness, inputs, 0, 0, length * 8, &stream)
                .unwrap();
            stream.synchronize().unwrap();
            println!(
                "Time taken before generating partial witness: {:?}",
                s.elapsed()
            );
        }
    }

    #[inline]
    pub(crate) fn replay(plan: &WitnessPlan) {
        let s = Instant::now();
        let stream = CudaStream::default();
        if let Some(streamed) = plan.streamed.as_ref() {
            Self::replay_streamed(streamed, &stream);
            if let Some(scatter) = plan.scatter.as_ref() {
                Self::launch_scatter(scatter, &stream);
            }
            stream.synchronize().unwrap();
            println!("  exec streamed witgen took: {:?}", s.elapsed());
            return;
        }
        assert!(
            !plan.graph_execs.is_empty(),
            "witness CUDA graph must be built before prove"
        );
        for graph_exec in &plan.graph_execs {
            graph_exec.launch(&stream).unwrap();
        }
        if let Some(scatter) = plan.scatter.as_ref() {
            Self::launch_scatter(scatter, &stream);
        }
        stream.synchronize().unwrap();
        println!("  exec cuda graph took: {:?}", s.elapsed());
    }

    /// Free the scatter `representative_map` after the last replay. It is not
    /// captured by the generator graph. Captured SoA addresses stay put.
    pub(crate) fn release_uncaptured_after_replay(
        plan: &mut WitnessPlan,
        buffers: &mut WitnessBuffers,
    ) {
        plan.scatter = None;
        let n = buffers.representative_map.len();
        if n == 0 {
            return;
        }
        let bytes = n * mem::size_of::<u32>();
        let t = Instant::now();
        device_synchronize().unwrap();
        let old = mem::replace(&mut buffers.representative_map, empty_device_alloc());
        drop(old);
        device_synchronize().unwrap();
        let (free, _) = memory_get_info().unwrap();
        println!(
            "Time taken to release_scatter_map ({:.2} MiB, free now {:.0} MiB): {:?}",
            bytes as f64 / (1024.0 * 1024.0),
            free as f64 / (1024.0 * 1024.0),
            t.elapsed()
        );
    }

    /// Drop captured graphs / streamed kernel lists. A live `CudaGraphExec`
    /// still references witgen SoA; NVIDIA forbids recycling that memory until
    /// the graph is destroyed. Prefer keeping the graph through prove (extra
    /// FRI malloc) so `cudaGraphExecDestroy` is not on the prove clock. Only
    /// call this mid-prove when SoA+FRI would not fit.
    pub(crate) fn release_after_replay(plan: &mut WitnessPlan) {
        plan.graph_execs.clear();
        plan.streamed = None;
        plan.scatter = None;
    }
}
