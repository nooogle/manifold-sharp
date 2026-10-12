// Copyright 2026 Lars Brubaker
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ConvexDilation.cs — NOT A PORT. manifold-rust dilates a non-convex solid by a convex
// tool in Minkowski.cs's middle branch: one hull per triangle, then CsgTree.BatchUnion
// 1000 hulls at a time. This file answers the same question with the same hulls but a
// different reduction, and it is divergence ledger entry 6 (docs/RUST_DIVERGENCES.md)
// — an ADDED entry point, reached only when a caller asks for it by name.
// `Minkowski.Compute` is not routed through it and does not know it exists, so every
// ported path still produces the Rust's bits.
//
// ── Why a different reduction ───────────────────────────────────────────────────
// BatchBoolean pops the largest-vertex mesh first, so within a batch the growing union
// is always one operand: the batch is a serial chain absorbing one group after
// another, and ~85% of a dilation's time is spent there with every core but one idle.
// The hulls are the cheap part. Measured on a 2642-triangle part: 13.3 s through the
// ported path against 2.96 s here, identical volume. A single-threaded balanced tree
// was NOT faster — the gain is the parallelism a balanced tree exposes, not the shape.
//
// ── The tree ────────────────────────────────────────────────────────────────────
//   1. Leaves: the solid on its own, then runs of LeafSize hull units in seed order,
//      a unit being one triangle or, when dilating, a convex patch whose one hull
//      replaces its triangles' (ConvexPatches.cs proves it lies in the dilation),
//      each leaf building its units' hulls as Minkowski.cs does (same vertex-sum
//      order) and unioning them through the CSG tree (the same BatchUnion
//      the ported path uses, on a batch small enough that its serial chain is short).
//      Hulls live only inside their leaf, so memory is bounded by the leaves in
//      flight, not by the triangle count.
//   2. Pairwise levels: node k of a level is the union of nodes 2k and 2k+1 of the
//      level below; an odd last node is carried up unchanged. Each level is one map.
//
// ── Determinism: the proof every parallel site carries ──────────────────────────
// The leaf map and every level map go through Progress.MaybeParMapCtProgress →
// Par.MaybeParMapCt, so
// the MANIFOLD_PARALLEL switch (ManifoldParallel.Enabled) governs them as it governs
// the other sites, and with it off this file runs sequentially. With it on, each
// worker writes result[i] for its own i and reads only the two input meshes (leaves)
// or the previous level's array, which is complete and never written again once its
// map has returned. The boolean engine is read once, before the tree, and every union
// in it - leaf and level alike - runs on that engine. The tree's
// shape is a function of the input mesh alone (patches are grown sequentially by exact
// predicates, ConvexPatches.cs), and every node's operands are
// fixed by index, so the booleans a parallel run performs are the booleans a
// sequential run performs, on the same inputs in the same operand order — the answer
// is the same bits whatever the threads do.
//
// The one exception is the one Minkowski.cs's header already documents for its hull
// map: each hull and each boolean mints mesh IDs from the process-global counter, and
// in parallel they are taken in worker order rather than index order. Those IDs are
// opaque handles that reach no geometry and no topology, and the closing
// InitializeOriginal replaces them with one fresh ID anyway, exactly as Minkowski.cs
// finishes. ParallelismTests.ConvexDilationGeometryIsBitIdenticalInParallel measures it.
//
// ── Erosion: the same tree, minus the solid ─────────────────────────────────────
// Minkowski.cs's inset branch computes A \ (boundary(A) ⊕ B) from the very same
// per-triangle hulls, finishing with a BatchBoolean Subtract of their batch unions from
// A (ConvexErosion.cs's header covers the sign convention). TryComputeErosion is that
// with this file's reduction: the hull leaves WITHOUT the solid leaf, reduced by the
// same tree, then one solid − union on the same engine. One routine (TryReduce) serves
// both, so the two cannot drift apart. The determinism argument above carries over
// unchanged: the leaf and level maps are the same maps over an index set that is a
// function of the triangle count alone, and the closing subtraction is one boolean run
// after the tree has returned, on fixed operands, outside any parallel map. Its
// agreement with Minkowski.Difference is on volume and genus, for the reason below.
// Unlike dilation it takes a convex solid (the ported erosion sweeps those too).
//
// ── Nested and crossing shells ──────────────────────────────────────────────────
// Shells that nest or cross with the same orientation wind 2 where they overlap, which
// the exact engine's unions are not defined for. That can only happen where two
// components' bounding boxes overlap, so when any pair does (HasOverlappingComponents, a
// sort and sweep on X), the robust engine's RebuildWithRule(Positive) first turns the
// solid into the union of its shells, and the tree reduces that; a rebuild that is not a
// clean manifold declines. No classification is attempted: a clean hollow or interlocked
// part rebuilds to the same solid. The rebuild is one sequential call before any map. For
// erosion the union is also the answer a user means: the raw sweep would carve an inner
// shell's boundary out of material the union keeps.
//
// ── Progress from inside the top unions ─────────────────────────────────────────
// Each union is one unit, so a top-level union that runs for seconds would hold the
// bar still. Levels of at most SubProgressMaxPairs unions (and erosion's closing
// subtraction) therefore hand each exact boolean a stage sink (BooleanStageProgress.cs)
// feeding one NodeProgress per level, which reports the level's finished nodes plus
// every running node's fraction as fractional units through ReportUnits. Monotone even
// with the level's unions in parallel: every stage and completion lands under the
// tracker's lock and only a value above the last one is reported; a node's fraction
// only grows and is replaced by its whole unit when it finishes, so no node exceeds
// its unit. Those levels advance the reporter's counter only after their map returns,
// so no Advance report can race the tracker. The sink is a side channel: the booleans
// compute the same bits (Union's remarks). Wider levels report per node as before.
//
// ── What it is not ──────────────────────────────────────────────────────────────
// Not bit-identical to Minkowski.Sum. Union is associative on the solid, not on the
// mesh: reducing the same hulls in a different order rounds intersection vertices
// differently and triangulates differently. Volume and genus agree to roundoff —
// ConvexDilationTests pins it. (An earlier measurement showed a 2.6e-7 relative
// drift on an 11916-triangle part; that was a DedupeEdges defect, shared with the
// C++ and fixed in both ports (EdgeOp.Dedupe.cs, manifold-rust 4a99dc4), dropping
// solid out of one union node — not the reduction order.)
// Triangle lists do not agree, which is why this is a separate entry point rather
// than a reroute.
//
// ── Degenerate components ───────────────────────────────────────────────────────
// Reducing the hulls in this order can leave something the serial batches did not:
// where two subtrees' hulls have faces that nearly coincide, their exact union can keep
// a collapsed, zero-volume shell beside the real solid (a 4-triangle sliver 40 mm long
// on a closed L-bracket; volume and the solid itself unchanged, genus -1 instead of 0).
// Every component of material in a dilation contains a whole copy of the tool, so a
// closing sequential pass after the dilation tree drops any outward-wound shell with less
// than half the tool's volume; inward-wound cavity shells, which can be arbitrarily small,
// are kept, and the pass runs only for a tool that contains the origin, the case in which
// the solid itself lies inside the dilation (ConvexDilation.Degenerate.cs). Erosion skips it: its components carry
// no such bound and can be genuinely thin. The pass leaves the mesh untouched, bit for
// bit, whenever nothing qualifies, so the determinism argument above is unaffected, and
// a cancel that lands during it still ends in Cancelled.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <summary>
	/// The dilation of a non-convex solid by a convex tool, reduced through a balanced,
	/// parallel union tree instead of <see cref="Minkowski.Sum"/>'s serial batches.
	/// </summary>
	/// <remarks>
	/// A fast path and nothing more: <see cref="TryCompute"/> answers false for every
	/// input outside the non-convex ⊕ convex case, and the caller then runs
	/// <see cref="Minkowski.Sum"/>. See the file header for the tree and its determinism
	/// argument.
	/// </remarks>
	public static partial class ConvexDilation
	{
		/// <summary>
		/// Hulls per leaf. Small enough that a leaf's BatchUnion chain is short, large
		/// enough that the leaves outnumber the cores on any part worth speeding up.
		/// </summary>
		private const int LeafSize = 16;

		/// <summary>
		/// The leaf and tree-level maps' parallel threshold. Each element is a whole
		/// boolean, so two of them already pay for the dispatch.
		/// </summary>
		private const int UnionParThreshold = 2;

		/// <summary>
		/// The widest tree level whose unions report from inside. The top levels hold the
		/// few long booleans the bar would otherwise stall in; below this the per-node
		/// units already move it often, and a tracker there would only multiply callbacks.
		/// </summary>
		private const int SubProgressMaxPairs = 8;

		/// <summary>
		/// The Minkowski sum of a non-convex <paramref name="solid"/> and a convex
		/// <paramref name="tool"/>, as a union of the solid and one hull per triangle.
		/// </summary>
		/// <param name="solid">The shape being grown. Must be non-convex.</param>
		/// <param name="tool">The structuring element. Must be convex.</param>
		/// <param name="token">
		/// The cancellation token, or null. A cancelled run answers <c>true</c> with an
		/// empty result carrying <see cref="Error.Cancelled"/>, so a cancelled caller does
		/// not go on to run the general path.
		/// </param>
		/// <param name="progress">
		/// The progress reporter, or null. One <see cref="Phase.Minkowski"/> unit per hull,
		/// per leaf and per tree node, plus one for the closing normals pass that
		/// <see cref="Progress.CompletePhase"/> spends, so a finished run lands on 1.0.
		/// Declines happen before the phase opens and report nothing. Under the parallel
		/// switch workers advance the shared counter concurrently; the reporter drops a
		/// report that would go backwards (see <see cref="ProgressReporter.Advance"/>).
		/// </param>
		/// <param name="result">The dilated solid when this returns true; empty otherwise.</param>
		/// <returns>True when this path applied; false when the caller must run
		/// <see cref="Minkowski.Sum"/>.</returns>
		public static bool TryCompute(
			ManifoldImpl solid,
			ManifoldImpl tool,
			CancelToken? token,
			ProgressReporter? progress,
			out ManifoldImpl result)
		{
			return TryReduce(solid, tool, false, token, progress, out result);
		}

		/// <summary>
		/// The Minkowski erosion of <paramref name="solid"/> by a convex
		/// <paramref name="tool"/> — <see cref="Minkowski.Difference"/>'s answer,
		/// A \ (boundary(A) ⊕ B) — as the solid minus the tree union of one hull per triangle.
		/// </summary>
		/// <remarks>
		/// Unlike <see cref="TryCompute"/> a convex solid is taken: the ported erosion sweeps
		/// a convex solid triangle by triangle too, so the tree reduces the very same hulls.
		/// (<see cref="ConvexErosion"/>'s closed form is faster still where it applies.)
		/// </remarks>
		/// <param name="solid">The shape being eroded.</param>
		/// <param name="tool">The structuring element. Must be convex.</param>
		/// <param name="token">The cancellation token, or null; as <see cref="TryCompute"/>.</param>
		/// <param name="progress">
		/// The progress reporter, or null; as <see cref="TryCompute"/>, plus one unit for the
		/// closing subtraction.
		/// </param>
		/// <param name="result">The eroded solid when this returns true; empty otherwise.</param>
		/// <returns>True when this path applied; false when the caller must run
		/// <see cref="Minkowski.Difference"/>.</returns>
		public static bool TryComputeErosion(
			ManifoldImpl solid,
			ManifoldImpl tool,
			CancelToken? token,
			ProgressReporter? progress,
			out ManifoldImpl result)
		{
			return TryReduce(solid, tool, true, token, progress, out result);
		}

		/// <summary>
		/// The one routine behind dilation and erosion: the same leaves and the same tree,
		/// with the solid as an extra leaf when growing and subtracted from the union when
		/// eroding.
		/// </summary>
		/// <param name="solid">The solid whose triangles are swept.</param>
		/// <param name="tool">The convex tool.</param>
		/// <param name="inset">True for erosion, false for dilation.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <param name="result">The result when this returns true; empty otherwise.</param>
		/// <returns>True when this path applied.</returns>
		private static bool TryReduce(
			ManifoldImpl solid,
			ManifoldImpl tool,
			bool inset,
			CancelToken? token,
			ProgressReporter? progress,
			out ManifoldImpl result)
		{
			ArgumentNullException.ThrowIfNull(solid);
			ArgumentNullException.ThrowIfNull(tool);

			result = new ManifoldImpl();

			if (Cancel.IsCancelled(token))
			{
				result = Boolean3Functions.CancelledImpl();
				return true;
			}

			// Empty, soup and errored operands are the general path's business; it has a
			// considered answer for each (a clone, a propagated status).
			if (solid.IsEmpty()
				|| tool.IsEmpty()
				|| solid.IsSoup
				|| tool.IsSoup
				|| solid.Status != Error.NoError
				|| tool.Status != Error.NoError)
			{
				return false;
			}

			// Exactly Minkowski.Compute's middle branch without the operand swap: convex ⊕
			// convex is one hull there and needs no help, and non-convex ⊕ non-convex is a
			// different algorithm.
			if ((!inset && solid.IsConvex()) || !tool.IsConvex())
			{
				return false;
			}

			// Shells that nest or cross make winding number 2 where they overlap, which the exact
			// engine's unions are not defined for (on Thingi10K 54229 the part unioned with itself
			// keeps 0.29 of its volume; the raw tree lost up to 1.5% of the dilation). They can
			// only do that where component boxes overlap, so such a solid is first rebuilt by the
			// robust engine as the union of its shells - a clean hollow or interlocked part
			// rebuilds to the same solid - and the tree reduces that. The rebuild is one
			// sequential call before any map, so the determinism argument above holds on its
			// output. It gets no progress reporter: its robust phases each end on 1.0, which a
			// caller mapping this call onto one bar reads as finished for the rest of the rebuild
			// and the whole tree (or the fallback sweep, if it declines). The bar sits at its
			// start through the rebuild instead; the token still reaches it.
			if (HasOverlappingComponents(solid))
			{
				RebuildsRun++;
				ManifoldImpl rebuilt = Robust.RobustFunctions.RebuildWithRule(solid, WindingRule.Positive, token, null);
				if (Cancel.IsCancelled(token))
				{
					result = Boolean3Functions.CancelledImpl();
					return true;
				}

				if (rebuilt.IsEmpty() || rebuilt.IsSoup || rebuilt.Status != Error.NoError)
				{
					return false;
				}

				solid = rebuilt;
			}

			// Hull units: convex patches plus single triangles when dilating (ConvexPatches.cs
			// proves each patch hull lies inside the dilation), one triangle each when eroding,
			// where a patch hull is not sound. Built sequentially before the phase opens.
			List<int[]>? units = ConvexPatches.Build(solid, inset ? 1 : (PatchSizeOverride ?? ConvexPatches.MaxPatchSize), token);
			if (units is null)
			{
				result = Boolean3Functions.CancelledImpl();
				return true;
			}

			int numUnits = units.Count;
			LastHullCount = numUnits;
			int numHullLeaves = (numUnits + LeafSize - 1) / LeafSize;
			// Dilation puts the solid in as leaf 0; erosion keeps it out of the union and
			// subtracts the union from it at the end instead.
			int solidLeaves = inset ? 0 : 1;
			int numLeaves = numHullLeaves + solidLeaves;

			// A binary reduction of L leaves performs exactly L - 1 unions, whatever the
			// carries, so the node count is known before the tree is built. Erosion adds one
			// unit for its closing subtraction.
			ulong total = (ulong)numUnits + (ulong)numLeaves + (ulong)(numLeaves - 1) + 1 + (ulong)(inset ? 1 : 0);
			Progress.BeginPhase(progress, Phase.Minkowski, total);

			// Read once so every node of the tree runs on the same engine even if a host
			// flips the process default while this is in flight.
			BooleanEngine engine = BooleanConfig.DefaultEngine();

			// When dilating, leaf 0 is the solid; every other leaf k is the union of the hulls
			// of triangles [(k-solidLeaves)*LeafSize, ...+LeafSize), built inside the leaf so
			// only the leaves in flight hold hulls, never all numUnits of them. Each worker
			// reads only the two input meshes and writes its own slot.
			ulong unitsDone = (ulong)numUnits + (ulong)numLeaves;
			ManifoldImpl[]? level = Progress.MaybeParMapCtProgress(numLeaves, UnionParThreshold, token, progress, leaf =>
			{
				if (leaf < solidLeaves)
				{
					return solid.Clone();
				}

				int start = (leaf - solidLeaves) * LeafSize;
				int count = Math.Min(LeafSize, numUnits - start);
				return LeafUnion(solid, tool, units, start, count, engine, token, progress);
			});

			while (level is not null && level.Length > 1)
			{
				// Per-level gate: a cancel that landed inside the last level's booleans comes
				// back as Cancelled leaves, which must not be unioned into a result.
				if (Cancel.IsCancelled(token))
				{
					break;
				}

				ManifoldImpl[] below = level;
				int pairs = below.Length / 2;
				ManifoldImpl[]? merged;
				if (progress is not null && pairs <= SubProgressMaxPairs)
				{
					// A top level: few unions, each long. They report from inside through
					// one tracker, and the level's units are advanced only once its map has
					// returned (the file header's progress section).
					NodeProgress tracker = new NodeProgress(progress, unitsDone, pairs);
					merged = Par.MaybeParMapCt(pairs, UnionParThreshold, token, node =>
					{
						ManifoldImpl union = Union(
							below[2 * node],
							below[(2 * node) + 1],
							OpType.Add,
							engine,
							token,
							fraction => tracker.Stage(node, fraction));
						tracker.Complete(node);
						return union;
					});
					progress.Advance((ulong)pairs);
				}
				else
				{
					merged = Progress.MaybeParMapCtProgress(pairs, UnionParThreshold, token, progress, node =>
						Boolean3Functions.BooleanDispatch(
							below[2 * node],
							below[(2 * node) + 1],
							OpType.Add,
							engine,
							token));
				}

				unitsDone += (ulong)pairs;

				if (merged is null)
				{
					level = null;
					break;
				}

				if ((below.Length & 1) == 1)
				{
					// The odd node rides up unchanged; it costs no boolean and no unit.
					Array.Resize(ref merged, pairs + 1);
					merged[pairs] = below[below.Length - 1];
				}

				level = merged;
			}

			// The closing check CancelToken.cs's invariant requires: a cancelled token can
			// never produce a NoError result, whichever stage the cancel landed in.
			if (level is null || Cancel.IsCancelled(token))
			{
				result = Boolean3Functions.CancelledImpl();
				return true;
			}

			ManifoldImpl outR = level[0];
			if (inset)
			{
				// Minkowski.cs's closing merge with only two operands: the solid minus the
				// swept boundary, on the engine the tree ran on.
				NodeProgress? tracker = progress is null ? null : new NodeProgress(progress, unitsDone, 1);
				outR = Union(solid, outR, OpType.Subtract, engine, token, tracker is null ? null : fraction => tracker.Stage(0, fraction));
				tracker?.Complete(0);
				progress?.Advance(1);
				if (Cancel.IsCancelled(token))
				{
					result = Boolean3Functions.CancelledImpl();
					return true;
				}
			}

			// Unions of hulls whose faces nearly coincide can leave a collapsed shell beside the
			// real solid. A dilation's every component holds a copy of the tool, so one
			// sequential pass drops anything smaller (ConvexDilation.Degenerate.cs); an
			// erosion has no such bound and is left as the tree made it.
			if (!inset && ContainsOrigin(tool))
			{
				outR = WithoutStrayComponents(outR, 0.5 * Math.Abs(tool.GetProperty(Property.Volume)));
				if (Cancel.IsCancelled(token))
				{
					result = Boolean3Functions.CancelledImpl();
					return true;
				}
			}

			// Minkowski.Compute's closing AsOriginal, so the two entry points hand back the
			// same kind of mesh: one fresh original ID, normals and coplanar faces set.
			outR.InitializeOriginal();
			outR.SetNormalsAndCoplanar();

			Progress.CompletePhase(progress);
			result = outR;
			return true;
		}

		/// <summary>
		/// One boolean of the tree on <paramref name="engine"/>, with <paramref name="stage"/>
		/// hearing its progress when the engine is the exact one.
		/// </summary>
		/// <remarks>
		/// <see cref="Boolean3Functions.BooleanDispatch"/> on <c>Exact</c> is an empty phase
		/// opening on a null reporter followed by the four-argument
		/// <see cref="Boolean3Functions.BooleanWithToken(ManifoldImpl, ManifoldImpl, OpType, CancelToken?)"/>,
		/// so routing the exact engine to the sink-taking overload computes the same bits.
		/// Other engines run through the dispatch unchanged and report nothing from inside.
		/// </remarks>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <param name="op">The operation.</param>
		/// <param name="engine">The tree's engine.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="stage">The stage sink, or null.</param>
		/// <returns>The boolean's result.</returns>
		private static ManifoldImpl Union(
			ManifoldImpl a,
			ManifoldImpl b,
			OpType op,
			BooleanEngine engine,
			CancelToken? token,
			Action<double>? stage)
		{
			if (stage is not null && engine == BooleanEngine.Exact)
			{
				return Boolean3Functions.BooleanWithToken(a, b, op, token, stage);
			}

			return Boolean3Functions.BooleanDispatch(a, b, op, engine, token);
		}

		/// <summary>
		/// How many times this thread has run the robust rebuild in <see cref="TryReduce"/>.
		/// Test-only evidence of which path a solid took: the rebuild runs on the calling
		/// thread before any map, so a caller reading it before and after one call sees
		/// exactly that call's rebuild.
		/// </summary>
		[ThreadStatic]
		internal static int RebuildsRun;

		/// <summary>
		/// Test and measurement knob, read on the calling thread: the dilation's patch size
		/// cap, or null for <see cref="ConvexPatches.MaxPatchSize"/>. 1 is the per-triangle tree.
		/// </summary>
		[ThreadStatic]
		internal static int? PatchSizeOverride;

		/// <summary>This thread's last run's hull count (patches plus single triangles).</summary>
		[ThreadStatic]
		internal static int LastHullCount;

		/// <summary>
		/// Whether two of the solid's connected components have overlapping bounding boxes -
		/// the only way its shells can nest or cross, and so the only case in which its winding
		/// number can leave 0 or 1.
		/// </summary>
		/// <remarks>
		/// Boxes are closed, so touching counts as overlapping; that only costs a rebuild.
		/// Sort and sweep on X: components in order of their minimum X, each compared with the
		/// ones that start before it ends, returning on the first overlap. Memory is linear in
		/// the component count. The worst case is every component spanning one X range (say
		/// 20k parallel rods): about n²/2 cheap compares, some 2e8, well under a second, and
		/// accepted as such.
		/// </remarks>
		/// <param name="solid">The solid to test.</param>
		/// <returns>True when some pair of component boxes overlaps.</returns>
		private static bool HasOverlappingComponents(ManifoldImpl solid)
		{
			int numVert = solid.NumVert();
			int numComponents = LabelComponents(solid, out List<int> component);
			if (numComponents <= 1)
			{
				return false;
			}

			Vec3[] min = new Vec3[numComponents];
			Vec3[] max = new Vec3[numComponents];
			bool[] seen = new bool[numComponents];
			for (int v = 0; v < numVert; v++)
			{
				int c = component[v];
				Vec3 p = solid.VertPos[v];
				if (!seen[c])
				{
					min[c] = p;
					max[c] = p;
					seen[c] = true;
					continue;
				}

				min[c] = new Vec3(Math.Min(min[c].X, p.X), Math.Min(min[c].Y, p.Y), Math.Min(min[c].Z, p.Z));
				max[c] = new Vec3(Math.Max(max[c].X, p.X), Math.Max(max[c].Y, p.Y), Math.Max(max[c].Z, p.Z));
			}

			int[] order = Enumerable.Range(0, numComponents).Where(c => seen[c]).OrderBy(c => min[c].X).ToArray();
			for (int i = 0; i < order.Length; i++)
			{
				int a = order[i];
				for (int j = i + 1; j < order.Length && min[order[j]].X <= max[a].X; j++)
				{
					int b = order[j];
					if (min[b].Y <= max[a].Y && min[a].Y <= max[b].Y
						&& min[b].Z <= max[a].Z && min[a].Z <= max[b].Z)
					{
						return true;
					}
				}
			}

			return false;
		}

		/// <summary>
		/// Builds the hulls of <paramref name="count"/> triangles starting at
		/// <paramref name="start"/> and unions them through the CSG tree — the same
		/// reduction Minkowski.cs applies to a whole batch, on a batch of one leaf.
		/// </summary>
		/// <param name="solid">The solid whose triangles are swept.</param>
		/// <param name="tool">The convex tool.</param>
		/// <param name="units">The hull units: patches and single triangles (ConvexPatches.Build).</param>
		/// <param name="start">The first unit of the leaf.</param>
		/// <param name="count">How many units the leaf holds, at least one.</param>
		/// <param name="engine">The engine every union in the tree runs on, read once up front.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null; one unit per hull.</param>
		/// <returns>The union of the leaf's hulls.</returns>
		private static ManifoldImpl LeafUnion(
			ManifoldImpl solid,
			ManifoldImpl tool,
			List<int[]> units,
			int start,
			int count,
			BooleanEngine engine,
			CancelToken? token,
			ProgressReporter? progress)
		{
			List<CsgLeafNode> children = new List<CsgLeafNode>(count);
			for (int i = 0; i < count; i++)
			{
				if (Cancel.IsCancelled(token))
				{
					return Boolean3Functions.CancelledImpl();
				}

				// The vertex sums in Minkowski.cs's order, so a single triangle's hull is the
				// same hull; a patch lists its distinct vertices in joining order.
				int[] unit = units[start + i];
				List<Vec3> simpleHull = new List<Vec3>(3 * unit.Length * tool.VertPos.Count);
				HashSet<int>? seen = unit.Length > 1 ? new HashSet<int>() : null;
				foreach (int tri in unit)
				{
					for (int k = 0; k < 3; k++)
					{
						int v = solid.Halfedge[(tri * 3) + k].StartVert;
						if (seen is not null && !seen.Add(v))
						{
							continue;
						}

						Vec3 aVert = solid.VertPos[v];
						foreach (Vec3 bVert in tool.VertPos)
						{
							simpleHull.Add(aVert + bVert);
						}
					}
				}

				ManifoldImpl hull = QuickHullFunctions.ConvexHull(simpleHull);
				progress?.Advance(1);
				if (count == 1)
				{
					return hull;
				}

				children.Add(new CsgLeafNode(hull));
			}

			// The same BatchUnion a CsgOp of these leaves would reach (identity transforms, so
			// flattening them changes nothing), called directly so it can take the engine.
			return CsgTree.BatchUnion(children, token, engine).GetImpl();
		}
	}

	/// <summary>
	/// The progress of one tree level's unions while they run, reported as fractional
	/// units: the level's finished nodes plus every running node's stage fraction.
	/// </summary>
	/// <remarks>
	/// Every stage and completion lands under one lock, and a value is reported only when
	/// it exceeds the last one reported, so however the workers interleave the sequence
	/// is strictly increasing. A running node's fraction only grows and is replaced by a
	/// whole unit when it finishes, so the value never exceeds the nodes' units and the
	/// level ends on exactly its node count.
	/// </remarks>
	internal sealed class NodeProgress
	{
		private readonly ProgressReporter reporter;
		private readonly double baseUnits;
		private readonly double[] running;
		private readonly object gate = new object();
		private int completed;
		private double lastReported;

		/// <summary>
		/// Starts tracking <paramref name="nodes"/> unions that begin after
		/// <paramref name="baseUnits"/> units of the phase.
		/// </summary>
		/// <param name="reporter">The phase's reporter.</param>
		/// <param name="baseUnits">The units finished before this level.</param>
		/// <param name="nodes">How many unions the level runs.</param>
		public NodeProgress(ProgressReporter reporter, ulong baseUnits, int nodes)
		{
			this.reporter = reporter;
			this.baseUnits = baseUnits;
			this.running = new double[nodes];
		}

		/// <summary>Node <paramref name="node"/> reached <paramref name="fraction"/> of its boolean.</summary>
		/// <param name="node">The node's index in the level.</param>
		/// <param name="fraction">Its completed fraction, 0 to 1.</param>
		public void Stage(int node, double fraction)
		{
			lock (this.gate)
			{
				this.running[node] = Math.Max(this.running[node], Math.Clamp(fraction, 0.0, 1.0));
				this.ReportLocked();
			}
		}

		/// <summary>Node <paramref name="node"/> finished its boolean.</summary>
		/// <param name="node">The node's index in the level.</param>
		public void Complete(int node)
		{
			lock (this.gate)
			{
				this.running[node] = 0.0;
				this.completed++;
				this.ReportLocked();
			}
		}

		private void ReportLocked()
		{
			double value = this.completed;
			foreach (double fraction in this.running)
			{
				value += fraction;
			}

			if (value > this.lastReported)
			{
				this.lastReported = value;
				this.reporter.ReportUnits(this.baseUnits + value);
			}
		}
	}

}
