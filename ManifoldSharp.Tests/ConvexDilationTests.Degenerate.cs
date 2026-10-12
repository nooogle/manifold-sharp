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

// ConvexDilationTests.Degenerate.cs — NOT A PORT; a C#-only adaptation test for
// ConvexDilation.Degenerate.cs (divergence ledger entry 6). The dilation tree's closing
// pass drops outward-wound shells with less than half the tool's volume, keeps cavity
// shells, runs only for a tool that contains the origin, and erosion skips it. The unit tests pin the rule on hand-built meshes; the
// end-to-end tests are a dilation that left a stray shell beside the real solid before
// the pass existed, and a near-closing cavity, a translated tool and an erosion to a
// genuinely thin rod that the pass must not touch, all with the ported sweep as the
// oracle.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
	public class ConvexDilationDegenerateTests
	{
		/// <summary>The bound the unit tests hand the pass: half of a unit tool's volume.</summary>
		private const double MinVolume = 0.5;

		/// <summary>
		/// A tetrahedron over the unit right triangle with its apex lifted by
		/// <paramref name="lift"/>: closed and manifold, and with volume lift / 6.
		/// </summary>
		private static Manifold Tetrahedron(double lift)
		{
			MeshGL64 mesh = new MeshGL64();
			mesh.NumProp = 3;
			mesh.VertProperties = new List<double> { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3, 0.3, lift };
			mesh.TriVerts = new List<ulong> { 0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3 };
			return Manifold.FromMeshGL64(mesh);
		}

		/// <summary>
		/// The L-shaped bracket closed by a 12-segment ball (grown, then shrunk back): a
		/// single genus-0 part whose rounded inside corner carries sliver triangles. Before
		/// the pass, dilating it by the same ball left a 4-triangle, zero-volume shell along
		/// the seam between its back face and its bottom edge.
		/// </summary>
		private static (Manifold Solid, Manifold Ball) ClosedBracket()
		{
			Manifold ball = Manifold.Sphere(2.0, 12);
			Manifold bracket = Manifold.Cube(new Vec3(40.0, 20.0, 6.0), false)
				.Union(Manifold.Cube(new Vec3(6.0, 20.0, 30.0), false));
			return (bracket.MinkowskiSum(ball, null, null).MinkowskiDifference(ball, null, null), ball);
		}

		/// <summary>
		/// A component smaller than the tool is dropped and the solid beside it kept, with the
		/// mesh's epsilon.
		/// </summary>
		[Test]
		public async Task AComponentSmallerThanTheToolIsDropped()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(10.0), false).Translate(Vec3.Splat(5.0));
			Manifold sliver = Tetrahedron(1e-9);
			await Assert.That(sliver.NumTri()).IsEqualTo(4).Because("the fixture's sliver must survive construction");

			Manifold mixed = Manifold.Compose(new List<Manifold> { cube, sliver });
			Manifold cleaned = Manifold.FromImpl(ConvexDilation.WithoutStrayComponents(mixed.AsImpl(), MinVolume));

			await Assert.That(ConvexDilation.LastDroppedComponents).IsEqualTo(1);
			await Assert.That(cleaned.Decompose().Count).IsEqualTo(1);
			await Assert.That(cleaned.Volume()).IsEqualTo(cube.Volume());
			await Assert.That(cleaned.AsImpl().Epsilon).IsEqualTo(mixed.AsImpl().Epsilon)
				.Because("the one kept part must carry the mesh's epsilon, not Decompose's unset -1");
			await Assert.That(cleaned.AsImpl().Tolerance).IsGreaterThanOrEqualTo(mixed.AsImpl().Tolerance);
		}

		/// <summary>
		/// A component at least the bound is kept however small it is beside the rest, and a
		/// mesh with nothing to drop comes back as the same instance.
		/// </summary>
		[Test]
		public async Task AComponentAsLargeAsTheBoundIsKept()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(10.0), false);
			Manifold unit = Manifold.Cube(Vec3.Splat(1.0), false).Translate(Vec3.Splat(20.0));
			Manifold mixed = Manifold.Compose(new List<Manifold> { cube, unit });

			ManifoldImpl before = mixed.AsImpl();
			ManifoldImpl after = ConvexDilation.WithoutStrayComponents(before, MinVolume);

			await Assert.That(ConvexDilation.LastDroppedComponents).IsEqualTo(0);
			await Assert.That(ReferenceEquals(after, before)).IsTrue().Because("a mesh with nothing to drop comes back untouched");
		}

		/// <summary>The closed bracket dilates to one piece, matching the ported sweep.</summary>
		[Test]
		public async Task TheClosedBracketDilatesToOnePiece()
		{
			(Manifold solid, Manifold ball) = ClosedBracket();
			await Assert.That(solid.Genus()).IsEqualTo(0);

			await Assert.That(solid.TryDilateByConvex(ball, null, null, out Manifold dilated)).IsTrue();
			Manifold reference = solid.MinkowskiSum(ball, null, null);

			await Assert.That(dilated.Status()).IsEqualTo(Error.NoError);
			await Assert.That(dilated.Decompose().Count).IsEqualTo(1);
			await Assert.That(dilated.Genus()).IsEqualTo(reference.Genus());
			double relative = Math.Abs(dilated.Volume() - reference.Volume()) / reference.Volume();
			await Assert.That(relative).IsLessThanOrEqualTo(1e-9);
		}

		/// <summary>
		/// A sealed cavity just wider than the tool survives. A 20 mm cube with a 4.2 mm cubic
		/// cavity, dilated by a ball of radius 2, keeps a void of 0.008 mm³ whose inward-wound
		/// shell is far below the tool's volume; the sweep keeps it, and so must the pass.
		/// </summary>
		[Test]
		public async Task ANearlyClosedCavitySurvives()
		{
			Manifold hollow = Manifold.Cube(Vec3.Splat(20.0), true).Difference(Manifold.Cube(Vec3.Splat(4.2), true));
			Manifold ball = Manifold.Sphere(2.0, 8);

			Manifold reference = hollow.MinkowskiSum(ball, null, null);
			await Assert.That(reference.Decompose().Count).IsEqualTo(2).Because("the sweep must keep the cavity, or the fixture tests nothing");

			// LastDroppedComponents is thread-static: read it before the first await, which may
			// resume on another thread.
			bool applied = hollow.TryDilateByConvex(ball, null, null, out Manifold dilated);
			int dropped = ConvexDilation.LastDroppedComponents;
			await Assert.That(applied).IsTrue();
			await Assert.That(dropped).IsEqualTo(0);
			await Assert.That(dilated.Decompose().Count).IsEqualTo(2);
			await Assert.That(dilated.Genus()).IsEqualTo(reference.Genus());
			double relative = Math.Abs(dilated.Volume() - reference.Volume()) / reference.Volume();
			await Assert.That(relative).IsLessThanOrEqualTo(1e-9);
		}

		/// <summary>
		/// A tool translated off the origin leaves the solid itself as a component of its own
		/// (Minkowski.Compute unions it in), smaller than the tool; the sweep keeps it, so the
		/// pass must not run.
		/// </summary>
		[Test]
		public async Task ATranslatedToolKeepsTheSolidsOwnComponent()
		{
			Manifold solid = ConvexDilationTests.LShape();
			Manifold tool = Manifold.Sphere(3.0, 8).Translate(new Vec3(20.0, 0.0, 0.0));

			Manifold reference = solid.MinkowskiSum(tool, null, null);
			await Assert.That(reference.Decompose().Count).IsEqualTo(2).Because("the solid must come out apart from the swept part, or the fixture tests nothing");
			await Assert.That(solid.Volume()).IsLessThan(0.5 * tool.Volume());

			await Assert.That(solid.TryDilateByConvex(tool, null, null, out Manifold dilated)).IsTrue();
			await Assert.That(dilated.Decompose().Count).IsEqualTo(2);
			double relative = Math.Abs(dilated.Volume() - reference.Volume()) / reference.Volume();
			await Assert.That(relative).IsLessThanOrEqualTo(1e-9);
		}

		/// <summary>
		/// Erosion keeps a genuinely thin rod. A bar 2r + 1e-6 across, eroded by a ball of
		/// radius r whose vertices lie on the axes, leaves a rod 1e-6 thick and 8e-12 in
		/// volume, which the sweep keeps; a thinness or volume test would take it for a stray.
		/// </summary>
		[Test]
		public async Task ErosionKeepsAThinRod()
		{
			const double r = 1.0;
			const double d = 1e-6;
			Manifold bar = Manifold.Cube(new Vec3(10.0, (2.0 * r) + d, (2.0 * r) + d), true);
			Manifold ball = Manifold.Sphere(r, 4);

			Manifold reference = bar.MinkowskiDifference(ball, null, null);
			await Assert.That(reference.Decompose().Count).IsEqualTo(1).Because("the sweep must leave the rod, or the fixture tests nothing");

			await Assert.That(bar.TryErodeByConvex(ball, null, null, out Manifold eroded)).IsTrue();
			await Assert.That(eroded.Decompose().Count).IsEqualTo(1);
			double relative = Math.Abs(eroded.Volume() - reference.Volume()) / reference.Volume();
			await Assert.That(relative).IsLessThanOrEqualTo(1e-9);
		}
	}
}
