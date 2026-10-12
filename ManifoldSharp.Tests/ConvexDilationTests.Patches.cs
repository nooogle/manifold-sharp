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

// ConvexDilationTests.Patches.cs — NOT A PORT; a C#-only adaptation test for
// ConvexPatches.cs (divergence ledger entry 6). The oracle is the same tree with one hull
// per triangle (patch size 1), the reduction main shipped before patches: the dilation
// is the same set either way, so volume and genus must agree to 1e-9.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
	public class ConvexDilationPatchTests
	{
		/// <summary>
		/// Relative volume slop between the patched and per-triangle trees. The two are the
		/// same set up to QuickHull's epsilon: a point within DefaultEps × extent of a face is
		/// dropped from a hull, so every hull can only shrink, never add material, and the
		/// trees differ only by such slivers plus the union order's roundoff.
		/// </summary>
		private const double VolumeTolerance = 1e-9;

		/// <summary>A 4 x 4 x 1 plate with a bore of radius 0.8 through its middle.</summary>
		private static Manifold DrilledPlate()
		{
			return Manifold.Cube(new Vec3(4.0, 4.0, 1.0), true)
				.Difference(Manifold.CylinderCentered(3.0, 0.8, -1.0, 16, true));
		}

		private static Manifold Shape(string shape)
		{
			return shape switch
			{
				"L-shape" => ConvexDilationTests.LShape(),
				"drilled" => ConvexDilationTests.DrilledPart(16),
				"drilled-plate" => DrilledPlate(),
				"frame" => ConvexDilationErosionTests.Frame(),
				"thin-wall" => ConvexDilationErosionTests.ThinWallDumbbell(),
				"thin-L" => ConvexDilationErosionTests.ThinL(),
				"hollow" => ConvexDilationTests.LShape().Scale(Vec3.Splat(2.0))
					.Difference(Manifold.Cube(Vec3.Splat(0.5), false).Translate(new Vec3(0.75, 0.75, 0.75))),
				_ => throw new ArgumentException(shape),
			};
		}

		private static (Manifold Result, int Hulls) Dilate(Manifold solid, Manifold tool, int? patchSize, bool skipGuard = false)
		{
			ConvexDilation.PatchSizeOverride = patchSize;
			ConvexPatches.SkipGuardForTests = skipGuard;
			try
			{
				if (!solid.TryDilateByConvex(tool, null, null, out Manifold result))
				{
					throw new InvalidOperationException("declined");
				}

				return (result, ConvexDilation.LastHullCount);
			}
			finally
			{
				ConvexDilation.PatchSizeOverride = null;
				ConvexPatches.SkipGuardForTests = false;
			}
		}

		/// <summary>
		/// Patches dilate to the per-triangle tree's solid: same genus, volume to 1e-9, and
		/// fewer hulls on every fixture with curved or bent faces.
		/// </summary>
		[Test]
		[Arguments("L-shape")]
		[Arguments("drilled")]
		[Arguments("drilled-plate")]
		[Arguments("frame")]
		[Arguments("thin-wall")]
		[Arguments("thin-L")]
		[Arguments("hollow")]
		public async Task PatchesMatchThePerTriangleTree(string shape)
		{
			Manifold solid = Shape(shape);
			Manifold ball = Manifold.Sphere(0.3, 8);
			(Manifold reference, int triangles) = Dilate(solid, ball, 1);
			(Manifold patched, int hulls) = Dilate(solid, ball, null);

			await Assert.That(hulls).IsLessThan(triangles).Because("no patch formed, so the comparison proves nothing");
			await Assert.That(patched.Status()).IsEqualTo(Error.NoError);
			await Assert.That(patched.Genus()).IsEqualTo(reference.Genus());
			double relative = Math.Abs(patched.Volume() - reference.Volume()) / reference.Volume();
			await Assert.That(relative).IsLessThanOrEqualTo(VolumeTolerance)
				.Because($"patched {patched.Volume()} against per-triangle {reference.Volume()} ({hulls} hulls for {triangles} triangles)");
		}

		/// <summary>
		/// A hull with a zero-area face is refused: every point is on such a face's plane,
		/// so without the strictly-below vertex it would pass as a supporting plane and then
		/// count every candidate triangle as separated from the hull.
		/// </summary>
		[Test]
		public async Task AZeroAreaHullFaceIsRefused()
		{
			Vec3 o = new Vec3(0.0, 0.0, 0.0);
			Vec3 x = new Vec3(1.0, 0.0, 0.0);
			Vec3 y = new Vec3(0.0, 1.0, 0.0);
			Vec3 z = new Vec3(0.0, 0.0, 1.0);
			List<Vec3> points = new List<Vec3> { o, x, y, z };

			// The unit tetrahedron, outward counterclockwise.
			Vec3[] tetra = { o, y, x, o, x, z, o, z, y, x, y, z };
			await Assert.That(ConvexPatches.FacesSupport(tetra, 4, points)).IsTrue();

			// The same faces plus one collapsed onto a segment.
			Vec3[] withSliver = { o, y, x, o, x, z, o, z, y, x, y, z, o, x, x };
			await Assert.That(ConvexPatches.FacesSupport(withSliver, 5, points)).IsFalse();
		}

		/// <summary>
		/// Sliver triangles stay out of patches. The closed bracket's rounded corner carries
		/// triangles whose height is below the solid's epsilon; their planes are set by
		/// roundoff, so condition (1) can certify a patch of them that is really flat (two
		/// of them on the bracket's back face once made a patch whose hull had volume
		/// 4.7e-14).
		/// </summary>
		[Test]
		public async Task SliverTrianglesStayOutOfPatches()
		{
			ManifoldImpl solid = ConvexDilationDegenerateTests.ClosedBracket().Solid.AsImpl();

			bool IsSliver(int tri)
			{
				Vec3 a = solid.VertPos[solid.Halfedge[3 * tri].StartVert];
				Vec3 b = solid.VertPos[solid.Halfedge[(3 * tri) + 1].StartVert];
				Vec3 c = solid.VertPos[solid.Halfedge[(3 * tri) + 2].StartVert];
				double twiceArea = Math.Sqrt(LinalgFunctions.LengthSquared(LinalgFunctions.Cross(b - a, c - a)));
				double longest = Math.Sqrt(Math.Max(Math.Max(LinalgFunctions.LengthSquared(b - a), LinalgFunctions.LengthSquared(c - b)), LinalgFunctions.LengthSquared(a - c)));
				return twiceArea < solid.Epsilon * longest;
			}

			List<int> slivers = Enumerable.Range(0, solid.NumTri()).Where(IsSliver).ToList();
			await Assert.That(slivers.Count).IsGreaterThan(0).Because("the fixture must carry sliver triangles, or it does not test the rule");

			List<int[]> units = ConvexPatches.Build(solid, ConvexPatches.MaxPatchSize, null)!;
			List<int> patched = units.Where(unit => unit.Length > 1).SelectMany(unit => unit).Where(IsSliver).ToList();
			await Assert.That(patched).IsEmpty().Because($"sliver triangles {string.Join(", ", patched)} joined a patch");
		}

		/// <summary>
		/// The guard keeps a bore open: a small ball leaves the drilled plate's hole (genus
		/// 1), and without the guard a patch hull spanning the bore's mouth fills it.
		/// </summary>
		[Test]
		public async Task TheGuardKeepsADrilledHoleOpen()
		{
			Manifold solid = DrilledPlate();
			Manifold ball = Manifold.Sphere(0.1, 8);

			(Manifold guarded, _) = Dilate(solid, ball, null);
			await Assert.That(guarded.Genus()).IsEqualTo(1);

			(Manifold unguarded, _) = Dilate(solid, ball, null, skipGuard: true);
			await Assert.That(unguarded.Genus()).IsEqualTo(0)
				.Because("without the guard the fixture must close the bore, or it does not test the guard");
		}
	}
}
