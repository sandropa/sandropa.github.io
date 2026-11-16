---
title: Neural Forecasting with Timestral
date: 2026-11-16
---

<figure>
{% include figures/timestral-fig1.svg %}
<figcaption><b>Figure 1.</b> Forecasting one neuron 29 seconds ahead, during a stimulus condition that none of the
models saw in training. The black line is the neuron's recorded activity, and the blue line is our forecast, with
shaded bands showing its uncertainty. The other lines are the forecasts of the published models. Our forecast follows
the sharp rise closely, while all published models underestimate it. This is one of the clearest examples; average
results are in Table&nbsp;1. (Neuron 40296 in the ZAPBench data; the forecast starts at frame 3380, about 51 minutes into
the recording.)</figcaption>
</figure>

<figure>
{% include figures/timestral-fig2-head.svg %}
<video id="f2-video" src="{{ '/assets/video/timestral-components-3d.mp4' | relative_url }}"
       poster="{{ '/assets/images/timestral-components.png' | relative_url }}"
       autoplay muted loop playsinline aria-label="Brain maps of the four components, slowly rocking side to side: each of the 71,721 neurons as a dot, red where it rises with the component and blue where it falls"
       style="width: 100%; display: block; mix-blend-mode: multiply;"></video>
<script>if (matchMedia("(prefers-reduced-motion: reduce)").matches) { var v = document.getElementById("f2-video"); v.removeAttribute("autoplay"); v.pause(); }</script>
{% include figures/timestral-fig2-time.svg %}
<figcaption><b>Figure 2.</b> The first four PCA components of brain-wide activity, which the model receives as
context next to each neuron. Top: every dot is one of the 71,721 neurons, seen from above while the brain
rocks gently from side to side to show its depth. Each neuron is coloured by its weight in the component: red
positive, blue negative, grey near zero. They explain 17%, 11%, 6% and 3% of the variance. Bottom: each component's
score over the two-hour recording. The bands mark the nine stimulus conditions, in order: gain,
dots, flash, taxis, turning, position, open loop, rotation and dark.</figcaption>
</figure>

<figure>
{% include figures/timestral-fig3.svg %}
<figcaption><b>Figure 3.</b> The whole two-hour recording of neuron 40296, the neuron from Figure&nbsp;1. The
recording consists of nine stimulus conditions, separated by vertical lines, and each one is split in time into training, validation and test data
(background colours). Taxis, the fourth stretch, is held out completely and is test data only.</figcaption>
</figure>

<figure>
{% include figures/timestral-fig4-head.svg %}
<video id="f4-video" src="{{ '/assets/video/timestral-method-taxis.mp4' | relative_url }}?v=11"
       poster="{{ '/assets/images/timestral-method-taxis-poster.png' | relative_url }}?v=11"
       autoplay muted loop playsinline aria-label="Animation: the forecast for neuron 40296 moving through the held-out taxis condition, with a prompt taken from the flash condition"
       style="width: 100%; display: block; mix-blend-mode: multiply;"></video>
<script>if (matchMedia("(prefers-reduced-motion: reduce)").matches) { var v4 = document.getElementById("f4-video"); v4.removeAttribute("autoplay"); v4.pause(); }</script>
<figcaption><b>Figure 4.</b> The forecast for neuron 40296 moving through the held-out taxis condition, which has
no training data. Top: the flash condition (left) and taxis (after the dotted line), coloured by training, validation
and test data. The black box is the moving window of the last 256 frames, and the blue line is the forecast with its
90% interval. The model recognises flash as the most similar known condition from brain activity alone, so the prompt
comes from flash's training data (dark, left) and stays the same for every forecast. Bottom: the model's input at
that moment: the prompt, a gap of missing values and the last 256 frames (black box), each with the neuron and the
PCA components (1 to 3 and 48 are shown). On the right, the forecast with its 50% and 90% intervals against what
actually happened (grey). The model also forecasts the components themselves (dotted blue); these are a by-product
and are not used. The animation shows every forecast moment of the taxis test period.</figcaption>
</figure>

<div class="algorithm">
<p class="algo-title"><b>Algorithm 1.</b> Zero-shot whole-brain forecasting</p>
<pre class="algo"><i>P</i> <span class="arr">⟵</span> top 48 PCA components of all neurons (training frames)

<b>for each</b> neuron and moment <i>t</i>:
    context <span class="arr">⟵</span> frames <i>t</i>−256 … <i>t</i>−1           <span class="c">benchmark input</span>
    <i>c</i>       <span class="arr">⟵</span> condition whose training <i>P</i> best matches context
    prompt  <span class="arr">⟵</span> last ≤ 768 training frames of <i>c</i>
    gap     <span class="arr">⟵</span> 32 missing frames
    input   <span class="arr">⟵</span> the neuron and <i>P</i> over prompt, gap, context
    <i>q</i>       <span class="arr">⟵</span> Timestral(input, 32 frames)  <span class="c">quantiles 5% … 95%</span>
    <b>return</b> the neuron's median and intervals from <i>q</i></pre>
</div>

<!-- PROVISIONAL: 500 random neurons, every 8th test window, 8 PCA components (results/baselines_compare.json, "pooled").
     To be replaced by the final run (1,000 fresh neurons, K chosen on validation). -->

| model | 0.9 s | 7 s | 29 s | average | taxis |
|---|---|---|---|---|---|
| **ours (zero-shot)** | **0.0181** | **0.0227** | **0.0254** | **0.0236** | **0.0269** |
| UNet | 0.0186 | 0.0230 | 0.0258 | 0.0239 | 0.0271 |
| TSMixer | 0.0190 | 0.0231 | 0.0258 | 0.0240 | 0.0271 |
| Time-Mix | 0.0190 | 0.0230 | 0.0259 | 0.0241 | 0.0271 |
| TiDE | 0.0190 | 0.0231 | 0.0259 | 0.0241 | 0.0571 |
| linear | 0.0197 | 0.0244 | 0.0277 | 0.0256 | 0.0285 |

<p class="caption"><b>Table 1.</b> Forecast error (mean absolute error, lower is better) 0.9, 7 and 29 seconds ahead,
averaged over all 32 steps, and on taxis, the held-out condition. 500 randomly chosen neurons; every model is scored on
the same neurons and the same moments.</p>

<figure>
{% include figures/timestral-fig4.svg %}
<figcaption><b>Figure 5.</b> Forecast error at every step ahead, from 0.9 to 29 seconds, for the same 500 neurons as
Table&nbsp;1 (mean absolute error, lower is better). Our forecast has the lowest error at every step. Errors grow with
the distance into the future for every model.</figcaption>
</figure>

{% include figures/timestral-table2.html %}
<p class="caption"><b>Table 2.</b> Forecast error per stimulus condition and forecast distance, ours / the published
model chosen above (mean absolute error, same 500 neurons as Table&nbsp;1). The first three columns are single steps; the
last is the average over all 32 steps, from 0.9 to 29 seconds. The lower error in each cell is in bold.</p>

| forecast quantile | 0.9 s | 7 s | 29 s | average |
|---|---|---|---|---|
| 5% | 5.4% | 4.7% | 4.7% | 5.0% |
| 10% | 11.0% | 10.2% | 10.4% | 10.6% |
| 25% | 26.7% | 26.2% | 27.2% | 27.0% |
| 50% (median) | 51.7% | 51.4% | 53.2% | 52.6% |
| 75% | 76.0% | 76.1% | 77.6% | 77.0% |
| 90% | 90.5% | 90.5% | 91.6% | 91.0% |
| 95% | 95.4% | 95.2% | 95.9% | 95.5% |

<p class="caption"><b>Table 3.</b> How well calibrated our forecasts are. Timestral forecasts quantiles, not a single
value. For each forecast quantile, the share of actual values that fell below it, at different distances into the
future (400 neurons); for a perfect forecaster the two are equal. Pairs of quantiles give prediction intervals: the
50%, 80% and 90% intervals contain the actual value 49.9%, 80.4% and 90.5% of the time. The published models give
single-value forecasts, without intervals.</p>
