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

// ConvexPatches.cs — NOT A PORT; part of divergence ledger entry 6 (ConvexDilation.cs).
// Stage B of the dilation tree: adjacent triangles are grouped into patches P so one
// hull, hull(P ⊕ B) = hull(P) ⊕ B, replaces |P| per-triangle hulls. DILATION ONLY.
//
// ── Why a patch hull is sound ───────────────────────────────────────────────────
// The dilation is solid ⊕ B. hull(P) ⊕ B ⊆ solid ⊕ B whenever H = hull(P) ⊆ solid
// (closed), and it covers every t ⊕ B for t in P, so swapping the |P| hulls for one
// leaves the union the same set - up to QuickHull's epsilon: a point within
// DefaultEps × extent of a face is dropped from a hull, so each computed hull, patch or
// per-triangle, can only shrink, never add material. The patched and per-triangle trees
// therefore drift apart only by such dropped slivers, which is why their volumes are
// compared at 1e-9 rather than bit for bit. A patch is accepted only when three exact checks
// prove H ⊆ solid, for a solid that is a closed, embedded, outward-oriented manifold
// with boundary surface S:
//   (1) Supporting faces: for every triangle t of P, every vertex of P is on or below
//       t's plane (Filtered.Orient3d, never Pos), and at least one is strictly below.
//       So t's plane supports H with H on the inward side, and t ⊂ ∂H; the strict
//       vertex makes t non-degenerate and gives H volume (a flat patch is refused -
//       a flat L's hull would cover the notch, and nothing below could see it).
//   (2) Guard: every triangle T of S outside P misses int H, shown by one face plane
//       of the computed hull with all three of T's corners on or above it (or T's own
//       plane with every P vertex on one closed side). The hull's faces are first
//       checked exactly to have every P vertex on or below them and at least one
//       strictly below, so each face is a real plane (a zero-area face would put every
//       point at zero and wave any T through) and int H lies in its open lower
//       halfspace whatever the hull's rounding.
//       A triangle no face separates (a real crossing or a tangency the test cannot
//       certify) refuses the patch. Candidates come from the solid's collider,
//       queried with the patch's box (H lies in it).
//   Then S ∩ int H = ∅: triangles outside P by (2), triangles of P by (1) (they lie
//   on ∂H). int H is open, convex and so connected, and misses S, so it is wholly
//   inside or wholly outside the solid. Just below the relative interior of any t in
//   P lies int H (t is 2D in a supporting face of the 3D body H) and also the solid
//   (t's outward normal). So int H ⊆ solid and H = closure(int H) ⊆ solid.
// Condition (1) implies every dihedral inside P is convex, but is stronger: a helical
// strip of convex edges fails it. Any failure puts the patch's triangles back as
// singletons, so doubt only costs speed.
//
// ── Sliver triangles stay singletons ────────────────────────────────────────────
// (1) is exact about each triangle's plane, but a sliver's plane - a triangle whose height
// is below the solid's epsilon - is set by roundoff, not by the surface. On a closed
// L-bracket two slivers lying on its flat back face (all four vertices within 1e-11 of
// the plane y = 20) each found the other's far vertex strictly below its tilted plane, so
// (1) and the guard accepted a patch whose hull had volume 4.7e-14: the flat patch (1)
// exists to refuse. Its hull's faces then nearly coincided with its neighbours', and the
// tree's union left a zero-volume shell (ConvexDilation.Degenerate.cs). So a sliver
// neither seeds nor joins a patch; it is swept as its own hull, as the per-triangle
// tree sweeps it.
//
// Why not erosion: the erosion tree subtracts ⋃ (t ⊕ B) = ∂A ⊕ B, and hull(P) ⊕ B is a
// superset that also contains int H ⊕ B - material the answer keeps. Not sound.
//
// ── Why the guard stays on the CPU ──────────────────────────────────────────────
// A GPU version of the guard was considered and dropped: patch building and the guard
// cost 3-7% of a patched run (Release, Thingi10K 40915/42041/40984), so a GPU filter
// could save at most a few percent.
//
// ── Determinism ─────────────────────────────────────────────────────────────────
// Sequential and a function of the mesh alone: seeds in triangle-index order, growth
// breadth-first across paired halfedges in halfedge order, each candidate taken or
// left by exact predicates. The unit list (patches and refused singletons) is in seed
// order, so the tree built on it is shaped by the input alone.

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp
{
	/// <summary>
	/// Groups a solid's triangles into provably interior convex patches for
	/// <see cref="ConvexDilation"/>'s dilation leaves. See the file header for the proof.
	/// </summary>
	internal static class ConvexPatches
	{
		/// <summary>The most triangles one patch may hold.</summary>
		internal const int MaxPatchSize = 16;

		/// <summary>
		/// Test-only, read on the calling thread: skip the guard (2), so a test can show a
		/// fixture the guard is what keeps open.
		/// </summary>
		[ThreadStatic]
		internal static bool SkipGuardForTests;

		/// <summary>
		/// The solid's triangles as hull units: each entry is a patch (triangles in the
		/// order they joined) or a single triangle. Entries are in seed-index order.
		/// </summary>
		/// <param name="solid">The solid, a valid closed manifold.</param>
		/// <param name="maxSize">The patch size cap; 1 gives one unit per triangle.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The units, or null when cancelled.</returns>
		internal static List<int[]>? Build(ManifoldImpl solid, int maxSize, CancelToken? token)
		{
			int numTri = solid.NumTri();
			List<int[]> units = new List<int[]>(numTri);
			if (maxSize <= 1 || solid.Collider is null || solid.Collider.NumLeaves() != numTri)
			{
				for (int t = 0; t < numTri; t++)
				{
					units.Add(new[] { t });
				}

				return units;
			}

			bool[] taken = new bool[numTri];
			List<int> patch = new List<int>(maxSize);
			List<int> patchVerts = new List<int>(3 * maxSize);
			Queue<int> frontier = new Queue<int>();
			for (int seed = 0; seed < numTri; seed++)
			{
				if (taken[seed])
				{
					continue;
				}

				if ((seed & 63) == 0 && Cancel.IsCancelled(token))
				{
					return null;
				}

				if (IsSliver(solid, seed))
				{
					taken[seed] = true;
					units.Add(new[] { seed });
					continue;
				}

				// A refused patch is released and regrown from the same seed at half the
				// cap, and always below the refused patch's size - growth that stopped short
				// of the cap would otherwise rebuild the same patch - down to a single
				// triangle. Growth is deterministic, so the smaller patch is a prefix of the
				// larger one.
				bool accepted = false;
				for (int cap = maxSize; cap > 1 && !accepted; cap = Math.Min(cap / 2, patch.Count - 1))
				{
					Grow(solid, seed, cap, taken, patch, patchVerts, frontier);
					accepted = patch.Count > 1 && Accept(solid, patch, patchVerts);
					if (!accepted)
					{
						for (int i = 1; i < patch.Count; i++)
						{
							taken[patch[i]] = false;
						}

						if (patch.Count <= 1)
						{
							break;
						}
					}
				}

				units.Add(accepted ? patch.ToArray() : new[] { seed });
			}

			return units;
		}

		private static void Grow(
			ManifoldImpl solid,
			int seed,
			int maxSize,
			bool[] taken,
			List<int> patch,
			List<int> patchVerts,
			Queue<int> frontier)
		{
			patch.Clear();
			patchVerts.Clear();
			frontier.Clear();
			patch.Add(seed);
			taken[seed] = true;
			AddVerts(solid, seed, patchVerts);
			frontier.Enqueue(seed);
			while (frontier.Count > 0 && patch.Count < maxSize)
			{
				int tri = frontier.Dequeue();
				for (int k = 0; k < 3 && patch.Count < maxSize; k++)
				{
					int paired = solid.Halfedge[(3 * tri) + k].PairedHalfedge;
					int neighbor = paired >= 0 ? paired / 3 : -1;
					if (neighbor < 0 || taken[neighbor] || IsSliver(solid, neighbor) || !Compatible(solid, neighbor, patch, patchVerts))
					{
						continue;
					}

					patch.Add(neighbor);
					taken[neighbor] = true;
					AddVerts(solid, neighbor, patchVerts);
					frontier.Enqueue(neighbor);
				}
			}
		}

		private static void AddVerts(ManifoldImpl solid, int tri, List<int> verts)
		{
			for (int k = 0; k < 3; k++)
			{
				int v = solid.Halfedge[(3 * tri) + k].StartVert;
				if (!verts.Contains(v))
				{
					verts.Add(v);
				}
			}
		}

		/// <summary>
		/// Whether a triangle's height is below the solid's epsilon (file header: its plane
		/// is roundoff, so it neither seeds nor joins a patch).
		/// </summary>
		private static bool IsSliver(ManifoldImpl solid, int tri)
		{
			Vec3 a = solid.VertPos[solid.Halfedge[3 * tri].StartVert];
			Vec3 b = solid.VertPos[solid.Halfedge[(3 * tri) + 1].StartVert];
			Vec3 c = solid.VertPos[solid.Halfedge[(3 * tri) + 2].StartVert];
			double twiceArea = Math.Sqrt(LinalgFunctions.LengthSquared(LinalgFunctions.Cross(b - a, c - a)));
			double longest = Math.Sqrt(Math.Max(
				Math.Max(LinalgFunctions.LengthSquared(b - a), LinalgFunctions.LengthSquared(c - b)),
				LinalgFunctions.LengthSquared(a - c)));
			return twiceArea < solid.Epsilon * longest;
		}

		/// <summary>
		/// Condition (1) kept incrementally: the candidate's corners are on or below every
		/// patch plane, and every patch vertex is on or below the candidate's plane.
		/// </summary>
		private static bool Compatible(ManifoldImpl solid, int candidate, List<int> patch, List<int> patchVerts)
		{
			foreach (int tri in patch)
			{
				for (int k = 0; k < 3; k++)
				{
					if (Side(solid, tri, solid.VertPos[solid.Halfedge[(3 * candidate) + k].StartVert]) == Sign.Pos)
					{
						return false;
					}
				}
			}

			foreach (int v in patchVerts)
			{
				if (Side(solid, candidate, solid.VertPos[v]) == Sign.Pos)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>Which side of triangle <paramref name="tri"/>'s outward plane a point is on, exactly.</summary>
		private static Sign Side(ManifoldImpl solid, int tri, Vec3 p)
		{
			return Filtered.Orient3d(
				solid.VertPos[solid.Halfedge[3 * tri].StartVert],
				solid.VertPos[solid.Halfedge[(3 * tri) + 1].StartVert],
				solid.VertPos[solid.Halfedge[(3 * tri) + 2].StartVert],
				p);
		}

		/// <summary>The strict half of condition (1), then the guard (2).</summary>
		private static bool Accept(ManifoldImpl solid, List<int> patch, List<int> patchVerts)
		{
			foreach (int tri in patch)
			{
				bool below = false;
				foreach (int v in patchVerts)
				{
					if (Side(solid, tri, solid.VertPos[v]) == Sign.Neg)
					{
						below = true;
						break;
					}
				}

				if (!below)
				{
					return false;
				}
			}

			List<Vec3> points = new List<Vec3>(patchVerts.Count);
			Box box = new Box();
			foreach (int v in patchVerts)
			{
				points.Add(solid.VertPos[v]);
				box.UnionPoint(solid.VertPos[v]);
			}

			ManifoldImpl hull = QuickHullFunctions.ConvexHull(points);
			int numFaces = hull.NumTri();
			if (hull.IsEmpty() || numFaces < 4)
			{
				return false;
			}

			Vec3[] faceCorners = new Vec3[3 * numFaces];
			for (int f = 0; f < numFaces; f++)
			{
				for (int k = 0; k < 3; k++)
				{
					faceCorners[(3 * f) + k] = hull.VertPos[hull.Halfedge[(3 * f) + k].StartVert];
				}
			}

			if (!FacesSupport(faceCorners, numFaces, points))
			{
				return false;
			}

			if (SkipGuardForTests)
			{
				return true;
			}

			bool clear = true;
			solid.Collider.CollisionsOne(box, 0, (_, tri) =>
			{
				if (!clear || patch.Contains(tri))
				{
					return;
				}

				if (!Separated(solid, tri, faceCorners, numFaces, points))
				{
					clear = false;
				}
			});

			return clear;
		}

		/// <summary>
		/// Whether every hull face is an exact supporting plane of the patch: every patch
		/// vertex on or below it and at least one strictly below. The strict vertex refuses
		/// a zero-area face, whose Orient3d is zero for every point and which would
		/// otherwise count every candidate triangle as separated.
		/// </summary>
		/// <param name="faceCorners">Three corners per face, counterclockwise from outside.</param>
		/// <param name="numFaces">How many faces.</param>
		/// <param name="points">The patch's vertices.</param>
		/// <returns>True when every face supports the patch.</returns>
		internal static bool FacesSupport(Vec3[] faceCorners, int numFaces, List<Vec3> points)
		{
			for (int f = 0; f < numFaces; f++)
			{
				bool below = false;
				foreach (Vec3 p in points)
				{
					Sign side = Filtered.Orient3d(faceCorners[3 * f], faceCorners[(3 * f) + 1], faceCorners[(3 * f) + 2], p);
					if (side == Sign.Pos)
					{
						return false;
					}

					below |= side == Sign.Neg;
				}

				if (!below)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// Whether a plane shows <paramref name="tri"/> missing the hull's interior: the
		/// triangle's own plane with every patch vertex on one closed side of it, or one hull
		/// face with all of the triangle's corners on or above it. Exact, and sufficient only:
		/// a pair the two tests cannot separate is treated as crossing.
		/// </summary>
		private static bool Separated(ManifoldImpl solid, int tri, Vec3[] faceCorners, int numFaces, List<Vec3> points)
		{
			Vec3 a = solid.VertPos[solid.Halfedge[3 * tri].StartVert];
			Vec3 b = solid.VertPos[solid.Halfedge[(3 * tri) + 1].StartVert];
			Vec3 c = solid.VertPos[solid.Halfedge[(3 * tri) + 2].StartVert];

			// The triangle's plane: int H lies in one open side when every hull vertex is on
			// the closed side, and the triangle lies in the plane. A degenerate triangle puts
			// every point at zero and falls through to the face planes.
			bool anyPos = false;
			bool anyNeg = false;
			foreach (Vec3 p in points)
			{
				Sign side = Filtered.Orient3d(a, b, c, p);
				anyPos |= side == Sign.Pos;
				anyNeg |= side == Sign.Neg;
			}

			if (anyPos != anyNeg)
			{
				return true;
			}
			for (int f = 0; f < numFaces; f++)
			{
				Vec3 f0 = faceCorners[3 * f];
				Vec3 f1 = faceCorners[(3 * f) + 1];
				Vec3 f2 = faceCorners[(3 * f) + 2];
				if (Filtered.Orient3d(f0, f1, f2, a) != Sign.Neg
					&& Filtered.Orient3d(f0, f1, f2, b) != Sign.Neg
					&& Filtered.Orient3d(f0, f1, f2, c) != Sign.Neg)
				{
					return true;
				}
			}

			return false;
		}
	}
}
