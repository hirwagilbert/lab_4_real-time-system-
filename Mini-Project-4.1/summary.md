# Summary: how jitter and overruns affected the loop

Section 4.1 splits a loop's deadline into a timing budget with an explicit margin for jitter and interference, and requires that budget to hold at worst-case execution time. Section 4.5 adds that a PLC scan cycle must be bounded and predictable, because stability depends on a sampling period that is both short and consistent. Our sweep (50 seeds, 20 s runs, T = 50 ms) shows the two hazards break these requirements in different ways.

With J = 0 and no overruns the real loop matched the baseline apart from the 4 ms computation time, confirming the implementation.

Jitter alone scatters the samples. The interval standard deviation grew to 4.1, 12.2 and 20.3 ms for J = 0.1T, 0.3T and 0.5T, but the mean actuation delay stayed near zero (4.1, 3.9 and -0.9 ms) because early and late samples cancel. The cost appears only in the spread, with a worst-case delay of about 34 ms, under one period.

Overruns behave differently. Because an overrun pushes back every following sample until the loop catches up, delay accumulates in one direction. At equal average disturbance (overrun probability 0.025, 0.075 and 0.125 with a mean overrun time of 2T) the mean delay was 8.2, 17.2 and 28.5 ms, the worst case reached 182 ms (3.6 periods), and up to 26 % of crossings were actuated more than one period late. In the stress run, with overruns of 10 to 40 periods, the loop fell permanently behind and 295 of 500 crossings were missed, the failure mode that Section 4.5 warns about.

Overruns were therefore the more damaging hazard: their output error was larger in every matched pair (relay controller 15.1, 22.4 and 28.6 % against 11.4, 16.1 and 22.8 % for jitter). The P rule was less sensitive in absolute terms but showed the same ordering. Of the two statistics, the share of intervals above 1.5T was more useful than the standard deviation, because it counts the large excursions a deadline cares about, but neither separates symmetric jitter from one-sided overruns; the signed mean delay and the worst case do.

Halving T with absolute jitter fixed doubled the relative jitter from 0.3T to 0.6T and raised the share of intervals above 1.5T from 6.3 % to 17.3 %, consistent with the sampling-regularity concern of Section 4.7: irregularity must be judged relative to the period.
