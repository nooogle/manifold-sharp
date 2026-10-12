# Deliberate divergences from manifold-rust

Per `CLAUDE.md`, producing identical floating-point values to
`manifold-rust` is the default and "close enough" is a bug; the entries below
are the deliberate exceptions. Each one is a case where the Rust behaviour is
either unreproducible in a managed runtime, unspecified in Rust itself, or a
provable defect, with the evidence that justified the choice made instead.
Trace-diff debugging against the Rust must expect these.

The plan predicted this file would stay empty. The first two entries arrived
with `linalg.rs` in Phase 1, and neither is an accuracy change: one replaces a
Rust hash that is not reproducible even across two runs of the same Rust binary,
and the other pins a tie that Rust explicitly leaves unspecified. The third
pinned an iteration order the Rust randomized per process; manifold-rust
`d3a5967` adopted the same pin, so all that remains of it is a throw where the
Rust panics. Those three are the contract's "genuinely unspecified Rust
behavior" clause — in each case there was no single Rust result to match.

The rest are additive, and most of them have since been taken by the Rust.
The fifth is a whole algorithm the Rust does not have, reachable only by name;
the sixth is a faster reduction for one Minkowski branch, again reachable only
by name, with the reporter hooks and the exact boolean's stage sink it needs;
the tenth is additive API, a progress reporter for CSG tree evaluation. Those
three are live, and none of them changes a specified numerical value or moves a
bit produced by a ported function.

Five are retired, their numbers, history and evidence kept so older references
resolve. The fourth (an appended `minkowski` progress phase with Minkowski
cancellation, a closing `CompletePhase` emit, and an emit that never runs a bar
backwards) was taken by manifold-rust `cd5853e` and `0729fde`. The seventh
repaired a defect the Rust and the C++ shared (QuickHull building non-convex
hulls); manifold-rust `a69e579` took the same fix. The eighth (a faster Auto
self-intersection pre-check) and the ninth (a faster coplanar cross-copy that
reports its own `coplanar overlaps` phase) were taken by manifold-rust `1ac0bc1`
and `a6d5097`. The eleventh (a cancel token for orientation repair) was taken by
manifold-rust `34787c6`. Each retired entry says what the Rust now does and
which Rust commit a trace diff needs.

## 1. `Vec2`'s hash is the plain field-order bit hash (2026-08-29)

**What differs:** `Vec2.GetHashCode` hashes the bit patterns of `X` then `Y` in
field order, the same shape as `Vec3` and `Vec4`. The Rust `impl Hash for Vec2`
does something else: it hashes `x` into the caller's hasher, hashes `y` into a
private `DefaultHasher`, then writes
`RandomState::new().build_hasher().finish() ^ (h2.finish() << 1)` — mixing in a
freshly seeded `RandomState` whose seed is drawn per call.

**Where:** `ManifoldSharp/Linalg/Vec2.cs`, `GetHashCode`. The Rust is
`src/linalg.rs` lines 2194-2206, whose comment describes the intent as
"XOR with shift — mirrors C++ std::hash specialization". The neighbouring
`Vec3` and `Vec4` impls, commented "Simpler hash that is consistent: just hash
all fields sequentially", are ported verbatim; only `Vec2` diverges.

**Why:** the Rust impl cannot be ported, because it is not a function of its
input. `RandomState::new()` seeds from thread-local random state, so the same
`Vec2` hashes differently on successive calls within one process, and the `x`
contribution written into `state` is then followed by an unrelated random word.
Reproducing that would import nondeterminism into a port whose entire premise is
bit-identical output. Nothing is lost by declining: the Rust type derives
`PartialEq` but never implements `Eq`, so this `Hash` impl cannot key a
`HashMap` or `HashSet` in Rust at all, and no call site uses it. On the C# side
`GetHashCode` must agree with `Equals`, which is bit-based, and the field-order
bit hash is the impl that does.

**Evidence:** the hash value is unreachable from any output. Per
`CLAUDE.md`'s dependency table every map in the port is documented
probe-only (`rustc-hash` → plain `Dictionary`/`HashSet`), so no iteration order
derived from a hash reaches a mesh. Equality, which *is* observable, stays
exactly bit-based — see `LinalgTests`.

## 2. Signed-zero ties in `MinF64` / `MaxF64` are pinned to .NET semantics (2026-08-29)

**What differs:** when the two arguments compare equal but differ in sign of
zero, `LinalgFunctions.MinF64` returns `-0.0` and `MaxF64` returns `+0.0`,
regardless of argument order. Rust's `f64::min` / `f64::max` do not specify a
result in that case, and what they actually produce depends on the target.

**Where:** `ManifoldSharp/Linalg/LinalgFunctions.cs`, `MinF64` / `MaxF64`, which
every Rust `.min()` / `.max()` in the linalg port routes through — so this
reaches `Min`, `Max`, `Clamp`, `MinElem`, `MaxElem`, `RotationQuatMat` and
`QSlerp`.

**Why:** Rust documents `f64::min` as following IEEE-754 `minNum` except for
signaling NaNs, and states outright that "if the inputs compare equal (such as
for the case of +0.0 and -0.0), either input may be returned
non-deterministically". That makes the tie a property of the backend, not of the
source: arm64's `FMIN`/`FMAX` are sign-of-zero aware, while x86's
`minsd`/`maxsd` resolve ties positionally and ignore the sign of zero, so the
answer there also depends on which operand LLVM put in which register. There is
no single "the Rust result" to match. This port therefore pins the tie to
.NET's `Math.Min`/`Math.Max`, whose signed-zero behaviour *is* specified and is
sign-of-zero aware in both operand orders — which is the arm64 Rust build, the
one the oracle lane compares against.

The NaN behaviour is not part of this divergence and is ported exactly:
`Math.Min`/`Math.Max` propagate NaN whereas Rust returns the non-NaN operand, so
`MinF64`/`MaxF64` screen both arguments for NaN before delegating. That screen
is the reason the wrappers exist at all.

**Evidence:** `LinalgTests.MinMaxF64SignedZeroTiesAreSignAware` pins the tie and
`LinalgTests.MinMaxF64NaNLoses` pins the NaN rule. The first implementation of
these helpers used `a < b ? a : b`, which returned `+0.0` for `MinF64(-0.0, 0.0)`
and `-0.0` for `MaxF64(0.0, -0.0)` — measured against Rust on arm64 during
review and corrected. Should the port ever need to match an x86 Rust build
bit-for-bit at a tie, this is the single place to change.

## 3. `Slice` throws where the Rust panics on an unpaired halfedge (2026-08-29; narrowed 2026-09-29)

**What differs:** where the Rust's `Impl::slice` walk reaches an unpaired `-1`,
widens it to a huge `usize` and panics on the next halfedge access, this port
throws `InvalidOperationException` at the same step. It is a stop-vs-stop
difference, and only off the manifold path: `Manifold.Slice` screens `IsSoup`
before calling, so only a direct `ManifoldImpl.Slice` on a soup impl reaches
either behaviour.

**Where:** `ManifoldSharp/FaceOp.Slice.cs`, `ManifoldImpl.Slice`, the guard in
the walk; the file header carries the same note at the code. The Rust is
`src/face_op.rs`, `ManifoldImpl::slice`.

**Why:** C# integer division makes the literal transcription unsafe rather
than merely different — `-1 / 3 == 0` would silently restart the walk at
triangle 0 and can spin forever, a wrong answer where the Rust stops.

**History — the seed order, retired as a divergence:** until 2026-09-29 this
entry was mainly about the loop seeding. The Rust held the plane-straddling
triangles in a `HashSet<usize>` with `RandomState` and seeded each loop from
`tris.iter().next()`, so its polygon order and each loop's start vertex changed
between two runs of the same binary (a differential harness saw five Rust runs
disagree on 58-65 of 73 slice cases, while agreeing on all 85 after
canonicalization, and every C# contour was bit-for-bit a Rust contour up to
rotation). This port pinned it: a `SortedSet<int>`, each loop seeded from
`Min`, the smallest triangle index still remaining. manifold-rust `d3a5967`
made the same choice — a `BTreeSet`, seeded from `tris.first()`, recorded as
its CPP_DIVERGENCES entry 7 against C++'s implementation-defined
`*unordered_set::begin()` — so the raw slice now matches the Rust bit for bit,
contour order and start vertex included.
`CrossSectionCtorTests.RawSliceContourOrderIsDeterministic` (the Rust's
`test_raw_slice_contour_order_is_deterministic`) pins it on a three-sphere
slice, and `RawSliceMatchesCppLerpBits` pins the crossing coordinates, which
since manifold-rust `a52bb8e` are C++'s `la::lerp` form in both ports.
`Project` never needed a pin — `assemble_halfedges` seeds from a `BTreeMap`.

## 4. Retired: the progress module gains a tenth `Phase` and a `CompletePhase` emit (2026-08-30; retired 2026-10-03)

**Not a divergence any more.** manifold-rust took all three items below.
`cd5853e` added `ProgressReporter::complete_phase` (plus the `Option`-aware
`progress::complete_phase`), which emits exactly 1.0 and parks the throttle, called
at the success point of the same six determinate robust phases and never on a
cancelled path (item 2), and made `emit` drop a fraction below the last one emitted
in the phase, reset by `begin_phase` (item 3). `0729fde` appended
`Phase::Minkowski = 9`, named "minkowski" (item 1), and threaded a cancel token and a
reporter through `minkowski_with_progress` the way `Minkowski.cs` does: the same unit
model (`work_units`: one unit per hull, per batch reduction, and the closing merge
spent through `complete_phase`), the same gates (entry, per hull-map element, per
batch or A-face, after the final merge), and the same whole bar on the empty-operand
exits. `a6d5097` then appended `Phase::CoplanarOverlaps = 10` with
`Phase::PIPELINE_ORDER` and `pipeline_position` (entry 9). Checked against the Rust
at `52c8772`: ids 0-10, every name, `Phase::ALL` (eleven long, index = id) and
`PIPELINE_ORDER` match `Phases.All`, `Phases.Name` and `Phases.PipelineOrder` value
for value, and `complete_phase` closes the same seven robust phases (five in
`intersection_graph.rs`, coplanar overlaps in `coplanar_clip.rs`, cells in
`cells.rs`) and Minkowski, never the three indeterminate ones. The tests are shared
1:1: `ProgressTests.EveryDeterminateRobustPhaseEndsAtExactlyOneInBothParallelModes`
and `RacingWorkersNeverSendTheBarBackwards` (the Rust's `progress_tests`), and the six
`MinkowskiProgressTests` (the Rust's `minkowski::progress_tests`).

What stays C#-only in the progress module is recorded under its own entries:
`ProgressReporter.ReportUnits` and `PhaseTotal` and the exact boolean's stage sink
(entry 6), the CSG tree's reporter (entry 10), and the `Phase.Minkowski` reports of
the two algorithms the Rust does not have, `ConvexErosion` and `ConvexDilation`
(entries 5 and 6). A progress-stream diff against the Rust needs a checkout at
`a6d5097` or later: before `cd5853e` every determinate phase stops short of 1.0 and
can report a lower fraction after a higher one, before `0729fde` Minkowski reports
nothing, and before `a6d5097` the coplanar cross-copy runs under "self
intersections". The published crate 0.15.0 and NuGet ManifoldRust 0.5.1 predate all
three; the oracle lane never observes progress.

**The entry as it stood before retirement** ("the Rust" below is manifold-rust
before `cd5853e`; the history and evidence are kept as written):

**What differs:** two additions to `progress.rs`'s port, both C#-only, both made
for the same caller.

1. `Phase` declares `Minkowski = 9`, appended after the Rust's nine, and
   `Phases.All` therefore has ten members instead of nine — so
   `Phases.FromId(9)` answers `Minkowski` where the Rust's `Phase::from_id(9)`
   answers `None`. The Rust's own nine ids and names are untouched, value for
   value and string for string.
2. `ProgressReporter.CompletePhase()` (plus the null-aware
   `Progress.CompletePhase`) emits the current phase at exactly 1.0,
   unconditionally. The Rust has no such method.
3. (2026-09-30) The emit drops a fraction below the last one emitted in the
   phase. Two workers whose increments land at 50 and 51 can reach the callback
   lock in the other order, and the Rust then reports 51 then 50; here the stale
   50 is dropped, so a consumer's stream never goes backwards. Equal fractions and
   null ones still pass, and `BeginPhase` resets the mark. Only which callbacks
   fire under contention changes, never a computed value
   (`ProgressOrderTests`).

**Where:** `ManifoldSharp/Progress.cs` (the enum, `Phases.All`, `Phases.Name`,
`ProgressReporter.CompletePhase`). `ManifoldSharp/Minkowski.cs` is the only thing
that *reports the appended phase*; the *emit* has six more callers, added
2026-08-30 — the robust engine's determinate phases, which had the same swallowed
tail. `IntersectionGraphBuild.cs` closes narrow phase, self intersections,
candidate points, registries and arrangements, and `Cells.cs` closes cells, each
at its success point only. The three indeterminate phases (`winding`, `assemble`
in `RobustFunctions.cs`, `exact boolean` in `Boolean3.Functions.cs`) deliberately
do not call it: with no total there is no bar to leave short, and `CompletePhase`
on one would only repeat the null fraction `BeginPhase` already emitted. The Rust
is `src/progress.rs`'s `#[repr(u32)] enum Phase`, whose ids are the FFI surface of
`manifold_rs_progress_phase_name`, and its `ProgressReporter`.

**Why the second one:** it repairs a defect the Rust shares. `Advance` emits only
when `done` crosses a step boundary, and `step` is `total / 100`, so every unit
after the last boundary is swallowed: at `total = 514` the final report is
510/514, and only when `total <= 100` — where `step` is 1 — does a finished phase
happen to land on 1.0. It bites every determinate phase, the boolean pipeline's
six included: on the union of two 96-segment spheres the narrow phase's last
report was 4600/4608, self intersections' and arrangements' 9200/9216, candidate
points' 312/314 and registries' 624/628, with cells (a variable-size advance per
edge group) stopping 0.000255 short. What made it look like a Minkowski-only
defect is that the boolean's *consumer* hides it — agg-sharp's
`BooleanProgressAdapter` keeps a high-water mark and closes each pairwise
operation's window itself, so a phase stopping at 0.9983 of its own bar is
invisible there, while a single-phase operation like Minkowski simply stops short
of full. Hiding is not fixing, so `CompletePhase` closes all seven.
Reporting is write-only from the kernel's point of view, so an extra emit cannot
reach a computed value; the risk it does carry — a straggler `Advance` reporting
a *lower* fraction after the 1.0 — is closed by parking the throttle inside
`CompletePhase`. It is deliberately not called on a cancelled or failed path,
where a full bar would be a false claim.

**Why the first one:** the Rust threads neither a progress sink nor a cancel token into
`minkowski`, and `cancel.rs`'s own deviation list says so — "C++ also threads ctx
into the non-Boolean entry points (`FromMeshGL`, `Smooth`, `LevelSet`, `Hull`,
`Minkowski`, `Refine`). Here only the boolean / CSG pipeline is cancellable at
all". A Minkowski erosion is the slowest operation this library performs (one
convex hull and one boolean per triangle of the eroded solid — minutes on a few
thousand triangles), and agg-sharp's morphological entry points had to be
synchronous purely because the kernel offered nothing to report or interrupt
with. Adding progress there needs a phase to report *as*: `ProgressReporter`
takes a `Phase`, and reusing one of the boolean's — "assemble", say — would name
the wrong pipeline in a user-facing string and break the monotonic-phase
contract `ProgressTests` asserts, since the batch reductions inside Minkowski run
whole booleans of their own. That contract is phases-in-pipeline-order: in the
Rust, ids 0-7 *are* pipeline positions, so its test checks ascending ids; here
appended ids are not (`CoplanarOverlaps`, id 10, runs third), so the port's
`Phases.PipelineOrder` lists the order once and the test compares positions in
it - the same claim on the Rust's phases, and the right one on appended ones.

Appending is what the enum's own doc comment already prescribes for growth
("new phases are appended rather than inserted"), and it is the minimal shape:
every Rust id keeps its Rust meaning, so a diff of reported ids on the nine
shared phases is unchanged, and only a consumer that enumerates `Phase::ALL` or
probes id 9 can tell the difference. No FFI consumes this enum from C# — the
Rust's `#[repr(u32)]` matters to `manifold-rs-ffi`, which this port does not
build.

**Evidence:** the whole progress module is write-only — the reporter cannot feed
a value back into the kernel, and `Minkowski.Compute` with a null reporter runs
the code that ran before either addition existed
(`Progress.MaybeParMapCtProgress` with a null reporter *is* `Par.MaybeParMapCt`,
and with a null token *is* `Par.MaybeParMap`). `MinkowskiTests`'
`DefaultParametersAreBitIdenticalToAnInstrumentedRun` pins that: the same fixture
run with defaults and with a live token plus a reporter produces bit-identical
vertex positions and halfedges.
`ALargeRunEndsAtExactlyOneInBothParallelModes` is the regression test for the
swallowed tail — it uses a total past 100, where the throttle's step exceeds 1,
and asserts the last fraction is 1.0 sequentially and in parallel.
`ProgressTests.EveryDeterminateRobustPhaseEndsAtExactlyOneInBothParallelModes` is
its counterpart for the six boolean phases, on a fixture sized so that every one
of them is both past 100 units and off a step boundary — which it also asserts
(each phase's second-to-last report is below 1.0), so a fixture that drifted into
landing on a boundary would fail rather than quietly stop proving anything.
Dropping any one of the six `CompletePhase` calls fails it, naming that phase.
`ProgressTests.PhaseIdsRoundTrip` reads `Phases.All` dynamically and so covers
the tenth member without being edited. (When these tests became ports, the six
Minkowski progress tests moved from `MinkowskiTests` to `MinkowskiProgressTests`,
and `ProgressOrderTests.RacingWorkersNeverSendTheBarBackwards` into
`ProgressTests`, one file per Rust test module.)

## 5. A closed-form convex erosion the Rust has no counterpart for (2026-09-01)

**What differs:** this port adds a second erosion algorithm.
`ManifoldSharp/ConvexErosion.cs` computes the Minkowski difference of a *convex*
solid in closed form — as the intersection of the solid's face halfspaces, each
plane pushed inward by the tool's support in that direction — where
`minkowski.rs` has exactly one erosion, the per-triangle sweep, and applies it to
convex and non-convex solids alike. `Manifold.TryConvexErosion` is the only way
in; `ManifoldSharp/Manifold.Shapes.cs` names it right below
`MinkowskiDifference`.

**What does not differ, and this is the point:** `Minkowski.Compute`,
`Minkowski.Sum`, `Minkowski.Difference`, `Manifold.MinkowskiSum` and
`Manifold.MinkowskiDifference` are untouched. Nothing is rerouted. Every ported
entry point still runs the ported algorithm and still produces the Rust's bits,
including for a convex solid, where the closed form would have been available and
is deliberately not taken. So this entry adds a capability rather than changing
an answer — the same shape as entry 4's appended phase, one level up.

**Where:** `ManifoldSharp/ConvexErosion.cs` (new file, not a port, its header
carries the derivation) and `Manifold.TryConvexErosion` in
`ManifoldSharp/Manifold.Shapes.cs`. The Rust is `src/minkowski.rs`, whose three
branches are transcribed in `ManifoldSharp/Minkowski.cs`; none of them is this.

**Why:** erosion is the only operation in this library with no cheap branch. The
Rust's cost model is explicit that convex ⊕ convex is one hull and milliseconds,
but *every* erosion — convex operands included — is one convex hull and one
boolean per triangle of the eroded solid. Measured on this port, in Release, on
an M-series mac: a 12-triangle box eroded by a 72-triangle ball is 15-20 ms, a
436-triangle cylinder is 184 ms, a 288-triangle sphere is 400 ms and a
4024-triangle sphere is 2.7 s. The closed form answers the same four in 0.01,
0.07, 1.0 and 29 ms — 100x to 2500x, and the gap widens with triangle count
because the sweep is superlinear in it and the closed form is two convex hulls
whatever the count.

That matters to one caller in particular. agg-sharp's `RoundAllEdgesObject3D` is
a morphological *opening* — erode by r, dilate by r — which is the uniform
rolling-ball fillet by construction. The dilation was always milliseconds and the
erosion was always the whole bill; on the box shape class the opening drops from
~16 ms to ~0.12 ms, about 130x, and on a part with a few thousand triangles it is
the difference between a progress bar and a cancel button and neither being
needed.

The mathematics is exact rather than approximate, which is why this is worth
having as more than a heuristic. A convex A *is* `{ x : n_i·x <= d_i }` over its
faces; erosion distributes over an intersection of halfspaces; and eroding one
halfspace by B slides its plane by B's support. So
`A ⊖ B = { x : n_i·x <= d_i - h_B(-n_i) }` with no tessellation of the offset
surface and no boolean anywhere. The sign is `h_B(-n_i)` because `Minkowski.cs`'s
erosion is `A \ (∂A ⊕ B)` — it sweeps B, not -B — so what it computes is
`{ x : x - B ⊆ A }`, and the closed form is written to answer the same question.
For the centred ball every real caller erodes with, `B = -B` and the distinction
is invisible; it is matched anyway so an asymmetric tool cannot silently answer
something else.

Two implementation choices are worth recording. The halfspace intersection is
built through the **polar dual** — with a point p strictly inside the result, a
facet of `hull{ n_i / c_i }` names the three planes meeting at a vertex — because
the obvious alternative, clipping plane by plane, is one boolean per face and
costs more than the sweep it replaces for anything past a prism. And the dual is
used *only* to enumerate those triples: each vertex is then solved in the primal
from the original `n` and `d`, so a box's corners land exactly on their planes
instead of round-tripping through `1/c`. That is what makes the 20-cube eroded by
a unit ball come out as volume 5832.0 exactly.

**Why it is a `Try` and not a reroute:** the closed form has a real applicability
gate, and a gate is a thing a caller should see rather than a thing hidden inside
an answer. It declines — returns false, and the caller runs the sweep — on: a
non-convex solid (the premise), a non-convex tool (not needed by the geometry,
needed by the promise: the sweep swaps its operands in that case and computes
something else), a tool that does not contain the origin, a vertex centroid that
is not strictly inside the eroded body, a degenerate plane triple, and any output
vertex that fails verification against every constraint it was built from.
Declining is always safe, because the sweep is the specification.

The tool-misses-the-origin decline is the interesting one and is a defect
*found* by this work rather than introduced by it. `A \ (∂A ⊕ B)` drops a point
of A exactly when its swept copy `x - B` *meets* the boundary. With the origin in
B that is the erosion, because `x` is itself in `x - B`: a swept copy that leaves
A has to cross out of it. With the origin outside B the swept copy can sail clear
over the boundary and land wholly outside, and the sweep keeps that point too — so
its answer stops being an erosion of anything. A 20-cube swept by a unit ball
centred at (2,0,0) comes back — in Rust and in this port alike — as a 32-triangle
solid of volume 5908 whose bounding box is still the whole cube. Being right and
being different is still a behaviour change, so the fast path stands down there
and the divergence is written here instead of shipped.

**Origin containment is the gate, and nothing weaker is.** The first version of
this tested the *pushes* instead — decline if `h_B(-n_i) < 0` for one of A's face
normals — on the reasoning that a negative push is a tool sticking out past that
face. That is a necessary condition and not a sufficient one, because it only ever
samples the directions A happens to have faces in. Review found the case that
walks through it: the same unit ball centred at (0.8,0.8,0), whose centre is 1.13
from the origin and which therefore reaches past none of a cube's six face planes.
It passed, the closed form answered the true erosion 5832, the sweep answered
5832.547, and the two disagreed in silence — the one outcome this whole design is
built to make impossible. The gate now tests the actual condition, origin ∈ B,
against the *tool's* own face planes: exact, since the tool is already known
convex, so a convex tool contains the origin precisely when every outward face
plane has a non-negative offset. It also subsumes the push test, since `0 ∈ B`
gives `h_B(-n) ≥ 0` in every direction, so that check is gone rather than kept
alongside.

**Where it is not exact.** The arithmetic is exact and beats the sweep at it — the
20-cube's 5832.0 has no rounding in it at all — and the volume agrees with the
sweep's to about 1e-14 relative, dense solids included. The triangulation does not
always: QuickHull discards dual points within its relative epsilon of an existing
facet, so on a finely tessellated solid a few redundant halfspaces are dropped. A
2048-triangle sphere eroded by a unit ball gives 4016 triangles against the sweep's
4020 (1152 triangles: 2544 against 2548). A 4.9e-7 volume gap measured here before
entry 7 was the sweep's own non-convex per-triangle hulls, not the dual hull.

**Evidence:** `ManifoldSharp.Tests/ConvexErosionTests.cs`, 22 cases, against two
oracles. The independent one enumerates every triple of offset face planes,
keeps the candidates where the tool actually fits — the *definition* of erosion,
not the plane-offset shortcut the production path is built on — and hulls the
survivors; it shares no code with the dual-hull construction it checks and is
cubic in the face count, so it could only ever be a test. Cube, tetrahedron and
icosahedron agree with it on volume and surface area to a relative 1e-6. The
second oracle is `Minkowski.Difference` itself, on the erosion and again on the
full opening (`erode` then `dilate`), which is the only thing Round All Edges
actually shows a user.

`TheGeneralDifferenceStillRunsTheSweep` is the regression test for the "nothing
is rerouted" claim above, and it is pinned on a shape where the two paths are
observably different meshes of the same solid — a 40x20x10 box, 36 triangles from
the sweep and 12 from the closed form, equal volumes — so a future edit that
quietly routed `MinkowskiDifference` through the fast path fails rather than
passing because the answers happen to match.
`AToolThatMissesTheOriginIsDeclined` runs both tool positions from the gate
paragraph above, the (0.8,0.8,0) regression included, and asserts the sweep's own
answer at each is unchanged. Neutering the gate fails both cases.

`AnAsymmetricToolErodesTheWayTheSweepDoes` is what makes the `h_B(-n)` sign a
measured claim rather than a written one. Every other fixture here uses a centred
ball, where `B = -B` and the reflection is invisible — flipping the sign failed
*nothing*, so the file header, the brute-force oracle's comment and this entry
were all asserting something no test could see. The tool reaches 1.0 in +x against
0.3 in -x and 0.8 in +z against 0.4 in -z, and the eroded 40-cube's bounding box
is asserted bit-exactly against the sweep's: X [-19, 19.7], Z [-19.2, 19.6]. With
the sign flipped those mirror and the test fails; it was run flipped to confirm
that, and it is the only test in the file that catches it.

`ADenseSolidAgreesWithTheSweepInVolume` pins the volume agreement on the
2048-triangle sphere at 1e-12 (measured 8e-15).

`ProgressIsReportedOnSuccessAndNotOnADeclineBeforeAnyWork` pins the reporting
contract: the appended `Phase.Minkowski` of entry 4, one unit per face of the
solid for the support pass plus one for everything after it, a bar that only rises
and ends at exactly 1.0, and silence on the decline that actually happens (a
non-convex solid, rejected before the phase opens) so the sweep's own `BeginPhase`
is the first thing a caller sees. The rare numeric declines *after* the phase has
opened leave it short on purpose — that is the price of having a seam mid-run at
all, and `ACancelRaisedDuringTheSupportPassIsObservedThere` is what it buys: a
token tripped from the second progress report is observed inside the support pass,
within 30 of the ~103 reports a completed pass emits.
`APreCancelledTokenAnswersCancelledRatherThanDeclining` pins the other end, that a
cancelled token comes back as *applied* — false would send a cancelled caller off
to run the minutes-long path it was cancelled out of — and the closing
`Cancel.IsCancelled` after the final hull is the twin of the one
`Minkowski.Compute` makes after its last `BatchBoolean`, so `CancelToken.cs`'s
invariant ("a cancelled token can never produce a `NoError` result") holds on this
path too.

## 6. A parallel union tree for non-convex ⊕ convex dilation, and erosion by a convex tool (2026-09-26)

**What differs:** this port adds a second reduction for the Minkowski sum of a
non-convex solid and a convex tool. `ManifoldSharp/ConvexDilation.cs` builds the
same per-triangle hulls `minkowski.rs` builds, then unions the solid and the hulls
through a balanced tree — leaves of 16 triangles, each building its own hulls and
unioning them through the CSG tree, then pairwise levels — with the leaf and level
maps going through `Progress.MaybeParMapCtProgress`, so `MANIFOLD_PARALLEL`
governs them.
`Manifold.TryDilateByConvex` is the only way in; it declines (returns false) for
anything but non-convex ⊕ convex. A solid with two or more components whose
bounding boxes overlap may have shells that nest or cross with the same
orientation, winding 2 where they overlap, which the exact engine's unions are not
defined for (Thingi10K 54229 and 54230 came out up to 1.5% small through the raw
tree). Such a solid is first rebuilt by the robust `RebuildWithRule(Positive)` into
the union of its shells, one sequential step before the maps, and the tree reduces
that; a rebuild that is not a clean manifold declines. A clean hollow or interlocked
part rebuilds to the same solid, so nothing is classified. `ConvexDilationNestedTests`
pins hollow, nested, interlocked, crossing, side-by-side overlapping and
ball-in-cavity-in-ball parts against the sweep of the rebuilt union (the raw erosion
sweep carves an inner shell's boundary out of kept material), and that the rebuild
runs exactly when component boxes overlap.

Erosion shares the routine (2026-09-30): `Manifold.TryErodeByConvex` builds the
same leaves without the solid leaf, reduces them through the same tree, and
subtracts the union from the solid in one boolean on the tree's engine — the
inset branch's A \ (boundary(A) ⊕ B) with the reduction swapped. It declines a
non-convex tool and every empty, soup or errored operand, and treats nesting as dilation does; it
takes a convex solid, which the ported sweep also sweeps triangle by triangle.
`Minkowski.Difference` and `Manifold.MinkowskiDifference` are untouched.
`ConvexDilationErosionTests` pins volume and genus against the sweep (an L, the
drilled part, a frame, a wall eroded through, a part eroded away, a cube), and
`ParallelismTests.ConvexErosionTreeGeometryIsBitIdenticalInParallel` the bit
identity.

Progress from inside the tree's top unions (2026-09-30): the exact boolean gains
C#-only overloads of `Boolean3Functions.BooleanWithToken`, `Boolean3.NewWithToken`
and `BooleanResultAssemble.BooleanResultWithToken` taking an optional
`Action<double>` stage sink, invoked with the constant marks in
`BooleanStageProgress.cs` at the gates that close each heavy stage, and
`ProgressReporter` gains `ReportUnits` for fractional units, and `PhaseTotal` so a
host reweighting the fraction can recover whole units (the tree's total follows its
patch count, not the triangle count). The sink reads nothing
and is handed only constants, so every boolean computes the same bits; the
overloads without it pass null. Only `ConvexDilation` passes one.

Convex patch hulls for dilation (2026-09-30): `ConvexPatches.cs` groups adjacent
triangles into patches P of up to 16 whose every triangle is a supporting face of
hull(P) and which no other surface triangle enters (exact `Filtered.Orient3d`
tests; the file header carries the proof that hull(P) then lies in the solid), and
the dilation leaves build one hull(P ⊕ B) per patch instead of one per triangle.
A refused patch is regrown at half size, down to a single triangle. The union is
the same set up to QuickHull's epsilon (points within DefaultEps × extent of a face
are dropped, so a hull can only shrink, never add material); volumes therefore agree
at 1e-9, not bit for bit. A hull face must have a patch vertex strictly below it, so a
zero-area face cannot wave the guard through. Erosion keeps
one hull per triangle: a patch hull there would carve kept material.
`ConvexDilationPatchTests` pins volume and genus against the per-triangle tree and
a drilled plate whose bore an unguarded patch fills. Release, parallel on, a ball
of 2% of the part: Thingi10K 40915 (4552 triangles) 4552 → 1901 hulls, 2.45 s →
0.94 s; 42041 (6162) 6162 → 3726, 2.35 s → 1.70 s; 40984 (18110) 18110 → 5595,
6.9 s → 2.8 s; patch building and guard 3-7% of the patched run.

Degenerate components (2026-10-12): the tree's order of unions can leave what the
serial batches did not, a collapsed, zero-volume shell beside the real solid where
two subtrees' hulls have nearly coincident faces (an L-bracket closed by a 12-segment
ball, dilated by the same ball: one 4-triangle sliver 40 mm long, the volume
unchanged, genus -1 where `Minkowski.Sum` gives 0; it appears with and without
patches). Every component of material in a dilation contains a whole copy of the
tool, so `ConvexDilation.Degenerate.cs` closes the dilation tree with one sequential
pass that drops any outward-wound shell with less than half the tool's volume, then
checks the token again. Cavity shells are wound inward and are kept whatever their size:
a sealed cavity just wider than the tool dilates to a void of almost nothing, which the
sweep keeps. The pass runs only for a tool that contains the origin: `Minkowski.Compute`
and the tree union the solid itself in, and with a translated tool it can come out as a
component of its own, smaller than the tool. Erosion skips the pass: its components carry no such bound (a bar 2r + 1e-6
across eroded by radius r leaves a real rod 1e-6 thick), and no thinness test tells
such a rod from a stray. The pass returns the tree's mesh untouched whenever there is
nothing to drop. `ConvexDilationDegenerateTests` pins the rule on hand-built meshes,
and the bracket, a near-closing cavity, a translated tool and the thin rod's erosion
against the sweep.

**What does not differ:** `Minkowski.Compute`/`Sum` and `Manifold.MinkowskiSum`
are untouched and still run the ported batches, so every ported entry point still
produces the Rust's bits. Like entry 5, this adds a capability rather than
changing an answer.

**Why:** `CsgTree.BatchBoolean` pops the largest-vertex mesh first, so each
1000-hull batch is a serial chain and most of a dilation's time runs on one core.
The tree exposes that work to the parallel switch; a single-threaded balanced tree
was measured and was not faster. Release, M-series mac, parallel on, a
128-triangle ball, drilled cubes: 6246 triangles 72.9 s → 9.0 s, 9826 triangles
92.8 s → 17.5 s, 11916 triangles 120.2 s → 21.7 s.

**Not bit-identical to the ported sum, by construction:** the same hulls unioned
in a different order round their intersection vertices differently, so triangle
lists differ. Volume and genus agree: on the three drilled cubes above the volumes
are bit-equal. (A first measurement showed up to 2.6e-7 relative drift; that was
a DedupeEdges defect shared with the C++ dropping solid out of one union node. It is
not a divergence: both ports fixed it the same way, manifold-rust in 4a99dc4 under
its own `docs/CPP_DIVERGENCES.md` entry 3, and `DedupeEdgesRegressionTests` pins it
here.) Sequential and parallel runs of the tree are bit-identical to each
other (`ParallelismTests.ConvexDilationGeometryIsBitIdenticalInParallel`), carrying
only the mesh-ID-order exception `Minkowski.cs`'s header already documents.

## 7. Retired: QuickHull decides "above a face" exactly (2026-09-30; retired 2026-10-03)

**Not a divergence any more.** This port's `QuickHull.Algo.cs` flood fill calls a
face visible from the apex, and `AddPointToFace` queues a point on a face, only when
the point is exactly above the face's three corners (`Filtered.Orient3d`;
`QuickHull.Exact.cs`), where the Rust and C++ (`quickhull.cpp:391`) read it off the
float distance to the face's stored plane (`d > 0.0`). That was a provable defect:
an apex collinear with an edge lies in the plane of both faces on it, rounding can
put one at +5.6e-17 and the other at 0, and the zero-area face coned to that horizon
edge, with its noise normal, leaves the hull non-convex (Thingi10K 63451 triangle
163 swept by Sphere(0.3, 8) left a vertex 0.58 outside a face; 22 of
CppNonConvexConvexMinkowskiSum's 368 hulls were non-convex). manifold-rust `a69e579`
made the same fix - `quickhull::is_above` / `is_face_above`, the same predicate at
the same two sites - recorded as its own `docs/CPP_DIVERGENCES.md` entry 11 against
the C++, so the two ports' hulls agree again. Both carry the containment tests
(`QuickHullContainmentTests`; the Rust's `test_hull_of_a_flat_triangle_swept_by_sphere_is_convex`
and `test_thingi641145_triangle109_swept_hull_is_convex`).

**History:** taken here first (2026-09-30). No ported expected value moved:
CppNonConvexConvexMinkowskiSum gives area 34.0630, genus 5 (was 34.0671), the
convex erosion's dense-sphere volume gap (4.9e-7, blamed on the dual hull in entry
5) closed to 8e-15, and `Minkowski.Sum` and `TryDilateByConvex` agree on Thingi10K
63451, 641145 and 287448. Agreement with the Rust on hulls of inputs holding
collinear points needs a checkout at `a69e579` or later (`CLAUDE.md`); the oracle
lane has no hull row.

## 8. Retired: the Auto engine's self-intersection pre-check runs faster (2026-10-02; retired 2026-10-03)

**Not a divergence any more.** manifold-rust `1ac0bc1` made the same three changes:
`boolean_dispatch_full` scans the operand with fewer triangles first (A on a tie);
`compute_self_intersections` runs its rows through the new `par::maybe_par_any_ct`,
parallel under `--features parallel` from `SELF_INTERSECT_PAR_THRESHOLD` = 1,000
triangles (this port's `Soup.SelfIntersectParThreshold`), with per-job scratch, a stop
on the first contact, and the same three answers (a hit is true even if a cancel
raced it, a cancel seen before any hit is `None`, otherwise false), the sequential
loop polling the token before every row; and `genuine_contact` settles a pair sharing
exactly one vertex whose other two corners lie strictly on one side of the other
triangle's plane, counting it in `vert_benign`. `Par.MaybeParAnyCt` is therefore a
port of `maybe_par_any_ct` now, not C#-only API, and the cached verdict and the
scan's counters agree with the Rust. The only difference left is the one every
parallel site has: rayon makes scratch per split job, `Parallel.For` per worker
task, and the scratch is discarded either way. The seven tests and the three
MatterCAD fixtures are shared 1:1 (`RobustSelfIntersectionScanTests`, the Rust's
`robust/self_intersection_scan_tests.rs`, fixtures byte-identical). Which operand
carries a cached verdict after an Auto boolean matches the Rust at `1ac0bc1` or
later; the published crate 0.15.0 and NuGet ManifoldRust 0.5.1 predate it, and no
oracle row reads a cached verdict.

**The entry as it stood before retirement** ("the Rust" below is manifold-rust
before `1ac0bc1`):

**What differs:** three changes, none of which changes which engine Auto picks.
`Boolean3Functions.BooleanDispatchFull` scans the operand with fewer triangles
first (A first on a tie), where the Rust scans A, then B.
`Soup.ComputeSelfIntersections` runs its per-triangle rows through
`Par.MaybeParAnyCt`, in parallel when `ManifoldParallel.Enabled` is set and the
mesh has at least 1,000 triangles; the Rust loop is sequential.
`Soup.GenuineContact` calls a pair that shares exactly one vertex benign when
t1's two other corners lie strictly on one side of t2's plane (the filtered exact
orient3d `RealSelfContact` already uses), before calling `RealSelfContact`.
`RealSelfContact` itself is unchanged.

**Why:** the scan decided most of a MatterCAD boolean's time: about 460 ms on a
21.6k-triangle part in Release, fully serial.

**Why the verdict cannot move:** the dispatch is `self_isect(a) || self_isect(b)`,
which is commutative, so the scan order changes only which operand pays for,
and caches, its scan. A large clean body cut by a small self-intersecting part
skips the body's scan. The parallel scan answers "some row has a genuine
contact". Any hit gives true, and a run where every row finished without a hit
gives false, which is what the sequential loop answers. A cancel still answers
true and caches nothing, unless a worker had already found a real contact. The
shortcut is sound because every point of t1 except the shared vertex is then
strictly off t2's plane, so the pair meets only in that vertex, which
`RealSelfContact` also calls benign. It mirrors the shortcut `RealSelfContact`
already has for t2's corners against t1's plane.

**What moves:** no geometry. Which operand carries a cached
`SelfIntersects` verdict after an Auto boolean can differ from the Rust. The
narrow-phase counters of the scan (local, never reported) differ.
`SelfIntersectionScanTests` (now `RobustSelfIntersectionScanTests`) pins the verdicts on three MatterCAD parts with the
switch on and off, the shortcut on synthetic pairs, and the dispatch both ways.
On the 21.6k-triangle part the scan went from about 500 ms to about 320 ms
sequential and to about 120-220 ms parallel on a loaded machine.

## 9. Retired: the robust engine's coplanar cross-copy runs faster and reports its own phase (2026-10-02; retired 2026-10-03)

**Not a divergence any more.** manifold-rust `a6d5097` made the same change in
`robust/coplanar_clip.rs`: `CoplanarClipRegion::prepare` does the polygon's side of
the work once, `clip` / `contains` reject a segment or point whose projected box
misses a positive-area polygon's box, `PrimIndex` answers "already present" from
probe-only hash sets, and the step reports `Phase::CoplanarOverlaps` (id 10,
"coplanar overlaps"), counted in regions and closed with `complete_phase`, only when
there are regions; `Phase::PIPELINE_ORDER` places it third and the order check
compares `pipeline_position`, as `Phases.PipelineOrder` / `PipelinePosition` do
here. Its 168-case harness (Robust and Auto, three ops, both builds) found every
MeshGL64 hash unchanged, and the frozen SHA-256 below is shared. Two differences in
layout, none in behaviour: the Rust keeps the step (`cross_copy_coplanar_regions`)
beside the region type in `coplanar_clip.rs`, where this port keeps
`CrossCopyCoplanarRegions` in `IntersectionGraphBuild.Types.cs` and the type in
`CoplanarClipRegion.cs`; and the Rust keys its indexes by (mesh, triangle) where this
port keys them by the destination list's reference, both probe-only. The Rust also
deleted `graph_geom.rs`'s per-call `clip_segment_to_polygon` and
`point_in_polygon_coplanar`, which this port's `GraphGeom.cs` still carries, uncalled
(`docs/FOLLOW_UPS.md`). The tests are shared 1:1:
`RobustCoplanarCrossCopyTests` (the Rust's `robust/coplanar_cross_copy_tests.rs`)
and `ProgressTests.RobustBooleanOverCoplanarFacesReportsPhasesInPipelineOrder`. A
progress stream matches the Rust's at `a6d5097` or later; geometry never differed.

**The entry as it stood before retirement** ("the Rust" below is manifold-rust
before `a6d5097`):

**What differs:** phase 3 of the intersection-graph build
(`IntersectionGraphBuild.Types.cs`, `CrossCopyCoplanarRegions`) prepares each
overlap polygon once (`Robust/CoplanarClipRegion.cs`) where the Rust's
`clip_segment_to_polygon` / `point_in_polygon_coplanar` recompute its normal,
axis, projection and orientation per call; rejects a primitive whose projected
box misses a positive-area polygon's box before the parametric clip; and answers
the "already present" test with a probe-only hash set per destination instead of
the Rust's linear scan. It also reports as a new, appended `Phase.CoplanarOverlaps`
(id 10, "coplanar overlaps"), counted in regions and closed with `CompletePhase`,
only when there are regions; the Rust runs the step silently, so progress kept
saying "self intersections" through it. Like entry 4's `Minkowski`, the id is
appended, so it is not the step's pipeline position: `Phases.PipelineOrder` places
it between self intersections and candidate points, and `ProgressTests` checks
phase order by that position, not by id (an id check fails on the first
cross-copy, 10 before 2 - `RobustBooleanOverCoplanarFacesReportsPhasesInPipelineOrder`).

**Why:** a 12-triangle cube resting on an 11,652-triangle self-touching body
(MatterCAD's Soccer Name Keychain) produced 375 coplanar regions with up to 740
primitives per cube triangle; phase 3 took 6.7 s of an 8.4 s robust boolean.

**Why no bit can move:** the prepared values are the ones the Rust computes, by
the same exact operations, and exact rationals do not round, so computing them
once changes nothing. The box reject only answers "empty" where the clip would:
a positive-area convex polygon is the intersection of its edge half-planes, so
anything the clip keeps lies in the polygon and its box (a zero-area polygon's
half-planes meet in an unbounded line, so the reject is off for it). The hash
probe answers exactly the scan's question (exact rational equality, prov equal,
either orientation) and appends still happen in scan order, so every primitive
list is element for element the Rust's. Progress is write-only.
`CoplanarCrossCopyTests` (now `RobustCoplanarCrossCopyTests`) pins a robust union over coplanar faces to the hash the
unoptimized code produced, sequentially and in parallel, and the new phase's
position and closing 1.0.

## 10. CSG tree evaluation takes a progress reporter (2026-10-02)

**What differs:** `CsgNode.EvaluateWithToken(CancelToken?, ProgressReporter?)` is
C#-only API; the Rust's `to_leaf_node` threads a cancel token and nothing else. The
reporter rides `ToLeafNode` / `CollectChildren` / `BatchUnion` / `BatchBoolean`
down to `SimpleBoolean`, which hands it to `BooleanDispatchWithProgress` in place
of `BooleanDispatch`. The tree's own steps (collapsing, Compose of disjoint groups,
the heap reduction) report nothing.

**Why:** a consumer's n-ary boolean (agg-sharp's batch path) ran with no progress
at all - a watched union, subtract or intersect showed one indeterminate step for
its whole duration, however many robust phases ran under it.

**How it reports:** each two-operand boolean the tree runs streams its own complete
phase sequence, restarting at its first phase and fraction 0, so phase order and
fractions are monotonic within one boolean, not across the tree. A consumer that
wants one bar keeps a high-water mark (agg-sharp's `BooleanProgressAdapter` does).
That holds under `ManifoldParallel` too: with a reporter, `BatchBoolean` runs each
round's pairs one after another instead of side by side (the Rust's `batch_boolean`
round parallelism, manifold-rust `fb1a52e`), since one `ProgressReporter` tracks one
phase at a time and two booleans streaming into it at once would interleave. The
pairs, their order and the result are the same either way; only the wall time of a
watched tree loses that round parallelism.

**Why no bit can move:** `BooleanDispatch` is `BooleanDispatchWithProgress(.., null)`,
so a null reporter is the code that ran before, and a live one is write-only from
the kernel's side (`Progress.cs`; retired entry 4). `CsgTreeProgressTests` runs the tree's union with and
without a reporter, sequentially and in parallel, and compares the meshes bit for
bit (seeing the coplanar overlaps phase arrive through it), and does the same for a
subtract through the public `EvaluateWithToken`.

## 11. Retired: orientation repair takes a cancel token (2026-09-30; retired 2026-10-03)

**Not a divergence any more.** manifold-rust `34787c6` added
`Manifold::repair_orientation_with_token` and `repair::plan_repair_with_token`:
`analyze_with_token` builds each shell's geometry and classification in one loop and
polls the token once per shell there and once per shell in the containment loop,
before the unclassified-shell skip, as `Repair.Analyze` does; a fired token gives a
`None` plan, and the manifold entry point turns it into `cancelled_impl()`, the empty
`Error::Cancelled` result, after the same empty-input shortcut. The tokenless entry
points pass `None`. The test is shared 1:1
(`RobustRepairTests.ARepairWithACancelledTokenReturnsCancelled`, the Rust's
`a_repair_with_a_cancelled_token_returns_cancelled`). The published crate 0.15.0 and
NuGet ManifoldRust 0.5.1 predate the API, so the oracle lane cannot call it.

**The entry as it stood before retirement** ("the Rust" below is manifold-rust
before `34787c6`):

**What differs:** `Manifold.RepairOrientationWithToken(CancelToken?)` and
`Repair.PlanRepairWithToken(tris, CancelToken?)` are C#-only API; the Rust's
`Manifold::repair_orientation` and `robust::repair::plan_repair` take no token. The
token is polled once per shell at the top of `Analyze`'s classification loop and
its containment loop (the shells x triangles part); once it fires, the plan is null
and the manifold is empty with `Error.Cancelled`, the shape every other cancelled
operation already returns.

**Why:** the analysis costs about shells x triangles, so a host repairing a soup of
many shells needs to be able to stop it, as it already can a boolean.

**Why no bit can move:** `RepairOrientation` and `PlanRepair` are the `WithToken`
bodies with a null token, which never cancels, so the tokenless path is the code
that ran before; a live token that does not fire only adds the polls.
`RobustRepairTests.ARepairWithACancelledTokenReturnsCancelled` pins the cancelled
result, the null plan, and a live token's repair equal to the tokenless one.
