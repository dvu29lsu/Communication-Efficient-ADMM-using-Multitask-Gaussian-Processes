CISS 2027 - focused experimental revision
25 November 2026

Main file: conference_101719.tex
Upload this project to Overleaf and select that main file. The project uses
IEEEtran conference format, 10-point body text, standard margins, and Times
text through T1 font encoding. It was compiled locally with Tectonic/XeTeX.
The supplied PDF is seven pages including references. Recheck pagination
with the selected Overleaf compiler before submission.

SCOPE
Sections IV and V are replaced by:
  sections/experimental_setup.tex
  sections/results_discussion.tex

The abstract and conclusion have limited changes to match the updated
evidence. The threshold multiplier condition is corrected to iota > 0.
The general introduction and method exposition are intentionally retained
for the next editing pass. Editorial comments are hidden in the review PDF.
Fonts and bibliography emphasis were normalized for the local compilation.
This is a review draft; it is not yet a six-page submission version.

IMPORTANT RESULT CHANGE
The table evaluates F(x) = sum_i f_i(x_i) + ||sum_i x_i||_1 at the terminal
primal variables. The supplied results sheet used the logged split objective
J(x,z) = sum_i f_i(x_i) + ||sum_i z_i||_1. These differ at finite residuals.
The same saved sweeps were reselected using F, subject to convergence,
T <= 2*T0, objective discrepancy <= 0.01%, and C < C0. Among eligible
settings, select the smallest query count; ties use discrepancy, iteration
count, iota, then alpha. Some selected settings and iteration counts changed.
No GP optimization experiments were rerun.

The objective reference is exact-update ADMM at the specified stopping
tolerances, not a certified optimum. The six baseline trajectories were
reproduced numerically and checked against their saved logs. Each selected
GP trajectory was checked for iteration count, query count, and terminal
residuals. Query counts include five exact initialization iterations (50
queries). Diagnostic oracle evaluations are excluded from communication
counts and do not justify wall-clock speed claims.

TABLES AND FIGURE
tables/main_results.tex                 Included in the main paper.
tables/selected_parameters.tex          Optional detailed settings; not included.
figures/verified_diagnostic.pdf         Included vector figure, generated from
                                       the corrected selected trajectories.

The supplied comparison and full verified selections are preserved in the
separate companion evidence files delivered with the revision. Original
input files and experiment code were not changed.

NEXT PASS
Use the separate revision-notes PDF for the six-page budget and remaining
method issues. In particular, align the rank-one LMC exposition with the
implemented rank-one factor plus learned diagonal task covariance; make
the burn-in/first surrogate iteration indexing explicit; describe the
observation-noise and projection conventions; and complete bibliography
metadata. The compiler still reports existing bibliography warnings for
rockafellar1998variational (missing year field) and williams2006gaussian
(both volume and number fields).
