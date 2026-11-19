---
title: Forecasting Brain Activity with Timestral
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

| model | 0.9 s | 7 s | 29 s | average | taxis |
|---|---|---|---|---|---|
| **ours (zero-shot)** | **0.0178** | **0.0225** | **0.0252** | **0.0235** | **0.0268** |
| UNet | 0.0183 | 0.0230 | 0.0258 | 0.0239 | 0.0275 |
| TSMixer | 0.0187 | 0.0231 | 0.0257 | 0.0240 | 0.0274 |
| Time-Mix | 0.0187 | 0.0230 | 0.0258 | 0.0241 | 0.0275 |
| TiDE | 0.0187 | 0.0231 | 0.0257 | 0.0241 | 0.0574 |
| linear | 0.0193 | 0.0243 | 0.0276 | 0.0255 | 0.0286 |

<p class="caption"><b>Table 1.</b> Forecast error (mean absolute error, lower is better) 0.9, 7 and 29 seconds ahead and
averaged over all 32 steps, over all stimulus conditions except taxis; the last column is taxis, the held-out condition,
averaged over all 32 steps. 1,000 randomly chosen neurons that were not used for
any choice of the method, every 8th test window; every model is scored on the same neurons and the same moments.</p>

<figure>
{% include figures/timestral-fig4.html %}
<figcaption><b>Figure 5.</b> Forecast error at every step ahead, from 0.9 to 29 seconds, for the same 1,000 neurons as
Table&nbsp;1 (mean absolute error, lower is better). The buttons choose the stimulus condition; the default is the average
over all conditions except taxis, where our forecast has the lowest error at every step. Errors grow with the distance
into the future for every model.</figcaption>
</figure>

{% include figures/timestral-table2.html %}
<p class="caption"><b>Table 2.</b> Forecast error per stimulus condition and forecast distance, ours / the published
model chosen above (mean absolute error, same 1,000 neurons as Table&nbsp;1). The first three columns are single steps; the
last is the average over all 32 steps, from 0.9 to 29 seconds. The lower error in each cell is in bold.</p>

| forecast quantile | 0.9 s | 7 s | 29 s | average |
|---|---|---|---|---|
| 5% | 5.6% | 4.6% | 4.7% | 4.9% |
| 10% | 11.3% | 10.1% | 10.4% | 10.5% |
| 25% | 27.4% | 26.7% | 27.5% | 27.3% |
| 50% (median) | 52.5% | 52.0% | 53.7% | 53.1% |
| 75% | 76.4% | 76.3% | 78.1% | 77.4% |
| 90% | 90.5% | 90.6% | 91.8% | 91.3% |
| 95% | 95.3% | 95.2% | 96.0% | 95.7% |

<p class="caption"><b>Table 3.</b> How well calibrated our forecasts are. Timestral forecasts quantiles, not a single
value. For each forecast quantile, the share of actual values that fell below it, at different distances into the
future (the same 1,000 neurons); for a perfect forecaster the two are equal. Pairs of quantiles give prediction intervals: the
50%, 80% and 90% intervals contain the actual value 50.1%, 80.8% and 90.9% of the time. The published models give
single-value forecasts, without intervals.</p>

<!-- SECTION B (4 frames of history), draft: text to come -->

<figure>
{% include figures/timestral-figB1.svg %}
<figcaption><b>Figure 6.</b> The same neuron as in Figure&nbsp;1, but now every model sees only the last 4 frames, 3.7
seconds (thick line before the dotted "now"), and forecasts the next 29 seconds. Top: a step up in the taxis condition
(frame 3701). Our forecast follows it closely, while the published models stay flat or react too late; our intervals are
wide, as they should be with so little to go on. Bottom: the moment of Figure&nbsp;1, about 5 minutes earlier. With so
little history no model foresees the large rise; the UNet, which sees the whole brain volume, gets partway, and our
intervals do not cover it.</figcaption>
</figure>

<figure>
{% include figures/timestral-figB.svg %}
<figcaption><b>Figure 7.</b> Three moments of neuron 40296, forecast from only the last 4 frames (dark line before the
dotted "now"). Left: the forecast with the recognised condition's recent training activity as the prompt. Middle: the
forecast with the 16 most similar moments from the training data. Right: their average, which is our forecast. Shading
shows each forecast's 50% and 90% intervals; the black line is what happened, and the red line is the UNet's forecast
from the same 4 frames (one released model), repeated in every panel for comparison. Top (taxis): the activity falls and stays
low; the condition prompt gets it right, the retrieved moments expect it to come back up, and the average lands in
between. Middle (flash): a sharp rise after 20 seconds; the retrieved moments time it right, the condition prompt is
late. Bottom (taxis): a step up; the condition prompt expects nothing and the retrieved moments too much, while the
average follows it closely. Averaging gives up some accuracy when one prompt is clearly right, and gains when
neither is.</figcaption>
</figure>
