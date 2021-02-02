# Outline

Intro to LSH schemes             

Review of control variates

Review of SRP + ICML paper

Consider control variates for ICML case with 1 extra vector

- preliminary Mathematica results show that you get a cubic instead of a quadratic in theta_{12} (makes it simpler to solve -> just take the root within (0,1) with quadratic formula)
- SRP proof showed that variance is <= rather than strict inequality ; control variates makes this a a strict inequality so long  cov( )^2 / var is not zero
can control variates be extended?

I suspect not -> because of two reasons (which we can put in the paper)

a) Control variates of the form 1_{e_is = e_js} cannot solely be used, i.e. if we look at pairs, eg theta_{e1, e2}, or any theta_{ei, ej}, we always get zero for Cov(Y, 1_{v_ei \neq v_es}) ; hence we can't do something similar to random projections

b) therefore always need to at least look at  control variates of the form 1_{v_{1s} = v_{eis}}.

But how can we choose them to ensure that we never get higher polynomial powers? suspect can only have at most one form of 1_{v_{1s} = v_{eis}}. ; but in that case, will the other control variates be fine (possibly should use Mathematica or Python to quickly explore this)

If there is an optimal choice ; write that

If not, then fall back to MLE, explain that for an algorithm even if we can always find 3-way similarities or n-way similarities, will have to solve higher degree polynomials unless we are lucky (.e.g SRP)

Hence cannot be extended like control variates by choosing extra vectors (at least not from this direction)

Plots for 4 -> is CV or SRP better?

Conclusion
 
# Others

- Work on getting the same cubic (Mathematica)
- Possible small scale simulations for some additional vectors