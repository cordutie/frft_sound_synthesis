Several videos and sound examples showcasing a series of sound examples built using the techniques discussed herein are included below. A brief explanation of each visual and sonic example is provided in this section.

<div style="margin-top: 20px;"></div>
<details>
<summary><span style="font-weight: normal; font-size: 1.5em; color: black">3.1. Time-Frequency Domain Rotation</span></summary>
<div style="margin-top: 20px;"></div>

<p>
The rotation property of the FrFT is well-studied; however, we decided to include an example in this research as it is fundamental to understanding its implications on sounds.

For this example, we considered a sinusoid with a frequency of \(10025\) Hz (exactly in the middle of the sampleable frequency domain) and computed both the spectrogram of its FrFT and the real part of its FrFT for values of \(\alpha\) ranging from \(0\) to \(1\). Note that the FrFT in this case is computed over the entire signal. Each video correspond to one of the transformation mentioned before.
</p>


<div style="margin-top: 20px;"></div>
<div style="text-align: center;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/wILsqh9GTIo?si=H7enGu43S8OpJFCR" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

</details>

<div style="margin-top: 20px;"></div>
<details>
<summary><span style="font-weight: normal; font-size: 1.5em; color: black">3.2. Sound Example Group 1: \(\alpha\)-Synthesis 🎧</span></summary>
<div style="margin-top: 20px;"></div>

<p>
In our first group of sound examples, we used Method 1 to generate audio from the real part of the FrFT of several sinusoids with various window lengths.

Specifically, one second of audio is generated from a sinusoid for each frequency in \(\{55, 220, 880\}\) Hz. These sounds are then processed using the FrFT with window sizes of \(\{0.5, 1\}\) seconds, hop sizes equal to half of the window sizes, and angles in \(\{0, 0.01, 0.05, 0.1, 0.25, 0.5\}\). Each video corresponds to one frequency and one window size setting, and all transforms are displayed sequentially with angles increasing as mentioned. Spectrum and spectrogram representations are available in separate videos.
</p>

<div style="margin-top: 20px;"></div>
<div style="text-align: center;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/Q5bxa2bTDeE?si=Gr53U0MIKvJqeNrw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

</details>

<div style="margin-top: 20px;"></div>
<details>
<summary><span style="font-weight: normal; font-size: 1.5em; color: black">3.3. Sound Example Group 2: \(\alpha\)-Synthesis + Angle Manipulation 🎧</span></summary>
<div style="margin-top: 20px;"></div>

<p>
In this group of sound examples, we use Method 1 by computing the FrFT of several sinusoids while manipulating the value of \(\alpha\) throughout the transform.

Specifically, one second of audio is generated from a sinusoid for each frequency in \(\{55, 220, 880\}\) Hz. These sounds are then processed using the FrFT with window sizes of approximately \(0.046, 0.092, 0.18,\) and \(0.32\) seconds, with hop sizes equal to half of the respective window sizes, and values of \(\alpha\) increasing linearly from \(0\) to \(0.5\). Each video corresponds to one frequency, with all transforms displayed sequentially as the window sizes increase according to the values mentioned. Spectrum and spectrogram representations are provided in separate videos.
</p>


<div style="margin-top: 20px;"></div>
<div style="text-align: center;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/B0pOn1DQwiY?si=zykswsdaMBzfefca" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

</details>

<div style="margin-top: 20px;"></div>
<details>
<summary><span style="font-weight: normal; font-size: 1.5em; color: black">3.4. Sound Example Group 3: \(\alpha\)-Filtering + Band Manipulation 🎧</span></summary>
<div style="margin-top: 20px;"></div>

<p>
In our third group of sound examples, we used Method 2 to filter two sinusoids with an \\alpha\)-band pass filter. These \(\alpha\)-band pass kernels are created by multiplying in the \(\alpha\)-domain with the spectrum of an impulse response of the form
$$
IR(t) = \exp(-0.5((tb)^2)) \cos(2\pi c t),
$$
where \(t\) corresponds to time, \(b\) to the bandwidth of the filter, and \(c\) to the center frequency of the filter. The filters are applied over time using fixed values of \(\alpha\) and bandwidth \(b\), while varying the center frequencies \(c\).

Specifically, two seconds of two sinusoids at frequencies \(220\) Hz and \(3520\) Hz are generated and \(\alpha\)-band pass filtered using the spectrum of impulse responses in the form of the equation above with \(b=1\), \(c\) increasing exponentially (base \(2\)) from  \(100\) to \(10000\), and values of \(\alpha\) in \(\{0.01, 0.05, 0.1, 0.25, 0.5\}\). The filtering is done using window sizes of approximately \(0.19\) and \(0.38\) seconds, with hop sizes equal to half of the respective window sizes. Each video corresponds to a specific frequency and window size setting, with all transforms displayed sequentially as the angles increase according to the specified values.
</p>

<div style="margin-top: 20px;"></div>
<div style="text-align: center;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/VmrND7O0xfU?si=QBjePY4ke_9fNndm" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

</details>

