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

// ConvexDilation.Degenerate.cs — NOT A PORT; part of divergence ledger entry 6
// (ConvexDilation.cs, whose header's "Degenerate components" section says why it exists).
// The dilation tree's closing pass: drop material shells smaller than the tool.
//
// ── The rule ────────────────────────────────────────────────────────────────────
// When the tool contains the origin, every component of material in solid ⊕ tool
// contains a whole translated copy of the tool, so an outward-wound shell (positive signed volume) with less volume than the tool
// is not part of the dilation; the tree's stray shells have essentially none. Such a
// shell is dropped when its volume is below half the tool's: the half leaves room for
// QuickHull's epsilon, which can only shrink a hull.
//
// Decompose splits shells, not material, and a cavity's shell is wound inward (negative
// signed volume). A cavity carries no such bound - a sealed cavity just wider than the
// tool dilates to a void of almost nothing, which the sweep keeps - so an inward shell is
// never dropped. That leaves a stray which happens to come out inward-wound in place
// (one 4-triangle shell of volume -2.4e-19 did, with patches off); keeping it is no worse
// than having no pass, where filling a real cavity would change the part. The rule needs no epsilon and no notion of thickness, which is the point: a
// geometric "is it flat?" test cannot tell a stray shell from a genuinely thin solid (a
// stray on a 40 mm bracket was 0.26 mm long and some 4e-8 thick by volume over area, as
// thick as a real rod can be), and an epsilon-based one would drop the real rod below.
//
// Minkowski.Compute and the tree union the solid itself in, which is only part of
// solid ⊕ tool when the tool contains the origin. With a tool translated off it, the
// solid can come out as a component of its own, smaller than the tool, and the sweep
// keeps it; so the pass runs only for a tool that contains the origin, which a centred
// ball does.
//
// ── Dilation only ───────────────────────────────────────────────────────────────
// An erosion's components carry no such bound: eroding a bar 2r + 1e-6 across by a ball of
// radius r leaves a rod 1e-6 thick, which the sweep keeps. So the erosion path never runs
// this pass. No erosion in the tests that motivated the pass left a stray behind.
//
// ── Determinism ─────────────────────────────────────────────────────────────────
// Sequential, after the tree, on its finished mesh. A mesh with nothing to drop returns
// the input instance untouched, so the bits the tree produced are the bits returned.
// Otherwise Decompose and Compose rebuild the kept components in Decompose's order, a
// function of the mesh alone.

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp
{
	public static partial class ConvexDilation
	{
		/// <summary>How many components this thread's last closing pass dropped.</summary>
		[ThreadStatic]
		internal static int LastDroppedComponents;

		/// <summary>
		/// <paramref name="mesh"/> without its outward-wound shells (positive signed volume)
		/// below <paramref name="minVolume"/>, or the same instance when it has none. Inward
		/// cavity shells are kept whatever their size (file header).
		/// </summary>
		/// <param name="mesh">The dilation tree's result.</param>
		/// <param name="minVolume">Half the tool's volume.</param>
		/// <returns>The kept components, or <paramref name="mesh"/> itself.</returns>
		/// <remarks>
		/// A mesh whose every component is below the bound comes back untouched: a dilation
		/// of a non-empty solid always holds a component at least the tool's volume, so that
		/// cannot be the tree's answer, and the pass does not invent an empty one.
		/// </remarks>
		internal static ManifoldImpl WithoutStrayComponents(ManifoldImpl mesh, double minVolume)
		{
			LastDroppedComponents = 0;
			if (mesh.IsEmpty() || LabelComponents(mesh, out _) <= 1)
			{
				return mesh;
			}

			List<Manifold> parts = Manifold.FromImpl(mesh).Decompose();
			List<Manifold> kept = parts.Where(part => !IsStray(part.AsImpl(), minVolume)).ToList();
			if (kept.Count == parts.Count || kept.Count == 0)
			{
				return mesh;
			}

			LastDroppedComponents = parts.Count - kept.Count;

			// Decompose builds its parts without an epsilon (-1), and composing a single part
			// only clones it, so set it again, floored at the tree's, and keep the tree's
			// tolerance: what the mesh would have carried had nothing been dropped.
			ManifoldImpl cleaned = Manifold.Compose(kept).AsImpl();
			cleaned.SetEpsilon(mesh.Epsilon, false);
			cleaned.Tolerance = Math.Max(cleaned.Tolerance, mesh.Tolerance);
			return cleaned;
		}

		/// <summary>
		/// Whether the convex tool contains the origin: no face has it strictly outside,
		/// exactly. Only then does solid ⊕ tool contain the solid, so that every component
		/// of material holds a copy of the tool. Minkowski.Compute, and the tree, union the
		/// solid in regardless; with a tool translated off the origin it can come out as a
		/// component of its own, smaller than the tool, which the sweep keeps.
		/// </summary>
		private static bool ContainsOrigin(ManifoldImpl tool)
		{
			Vec3 origin = new Vec3(0.0, 0.0, 0.0);
			for (int tri = 0; tri < tool.NumTri(); tri++)
			{
				Vec3 a = tool.VertPos[tool.Halfedge[3 * tri].StartVert];
				Vec3 b = tool.VertPos[tool.Halfedge[(3 * tri) + 1].StartVert];
				Vec3 c = tool.VertPos[tool.Halfedge[(3 * tri) + 2].StartVert];
				if (Filtered.Orient3d(a, b, c, origin) == Sign.Pos)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// A shell wound outward (positive signed volume) but smaller than the bound. A cavity's
		/// shell is wound inward, its signed volume negative, and is never a stray: a cavity can
		/// legitimately shrink to almost nothing.
		/// </summary>
		private static bool IsStray(ManifoldImpl shell, double minVolume)
		{
			double volume = shell.GetProperty(Property.Volume);
			return volume >= 0.0 && volume < minVolume;
		}

		/// <summary>
		/// Labels each vertex with its connected component, joined along forward halfedges.
		/// </summary>
		/// <param name="mesh">The mesh to label.</param>
		/// <param name="component">Per vertex, its component's index.</param>
		/// <returns>The number of components.</returns>
		private static int LabelComponents(ManifoldImpl mesh, out List<int> component)
		{
			int numVert = mesh.NumVert();
			DisjointSets sets = new DisjointSets((uint)numVert);
			foreach (Halfedge halfedge in mesh.Halfedge)
			{
				if (halfedge.IsForward())
				{
					sets.Unite((uint)halfedge.StartVert, (uint)halfedge.EndVert);
				}
			}

			component = new List<int>(numVert);
			for (int i = 0; i < numVert; i++)
			{
				component.Add(0);
			}

			return sets.ConnectedComponents(component);
		}
	}
}
