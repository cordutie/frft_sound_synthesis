---
layout: default_no_margin
title: Fractional Fourier Sound Synthesis
---

[Back](./)

This table presents several examples in which various sound sources are transformed using the Fractional Fourier Transform (FrFT) with differing parameter settings. All audio materials originate from original recordings created by the first author for a previous project.

<style>
  .texture-grid {
    display: grid;
    grid-template-columns: repeat(11, 200px);
    gap: 1px;
    margin: 0 auto;
    max-width: 2200px;
    border-collapse: collapse;
    background-color: rgba(0, 0, 0, 0.05);
  }

  .texture-grid > div {
    background-color: white;
    border: 1px solid rgba(0, 0, 0, 0.08);
    padding: 10px;
    text-align: center;
  }

  .span-text-angle {
    grid-column: span 3;
    background-color: #f0f0f0;
    font-weight: bold;
  }

  .span-text-head {
    grid-column: span 2;
    background-color: #f0f0f0;
    font-weight: bold;
  }

  .sub-header {
    font-weight: bold;
    background-color: #e8e8e8;
  }

  audio {
    width: 180px;
  }

  .sound-label {
    font-weight: bold;
    text-transform: lowercase;
  }
</style>

<div class="texture-grid">
  <!-- Header Row 1 -->
  <div class="span-text-head "> </div>
  <div class="span-text-angle">Angle = 0.01</div>
  <div class="span-text-angle">Angle = 0.05</div>
  <div class="span-text-angle">Angle = 0.1</div>

  <!-- Header Row 2 -->
  <div> Sound Description</div>
  <div> Original Sound   </div>
  <div class="sub-header">ws = 2048</div>
  <div class="sub-header">ws = 4096</div>
  <div class="sub-header">ws = 8192</div>
  <div class="sub-header">ws = 2048</div>
  <div class="sub-header">ws = 4096</div>
  <div class="sub-header">ws = 8192</div>
  <div class="sub-header">ws = 2048</div>
  <div class="sub-header">ws = 4096</div>
  <div class="sub-header">ws = 8192</div>

  <!-- <div class="sound-label">Cacophonic melody played on an electric bass through a reverb pedal.</div>
  <div><audio controls src="./assets/audios/bass_cacophonic_original.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_cacophonic_angle_0.01_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_cacophonic_angle_0.01_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_cacophonic_angle_0.01_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_cacophonic_angle_0.05_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_cacophonic_angle_0.05_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_cacophonic_angle_0.05_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_cacophonic_angle_0.10_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_cacophonic_angle_0.10_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_cacophonic_angle_0.10_ws_8192.mp3"></audio></div> -->
  <!-- <div class="sound-label">Electric bass processed through a pitch-shifting delay (adding fifths in every iteration)</div>
  <div><audio controls src="./assets/audios/bass_delay_original.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_delay_angle_0.01_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_delay_angle_0.01_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_delay_angle_0.01_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_delay_angle_0.05_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_delay_angle_0.05_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_delay_angle_0.05_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_delay_angle_0.10_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_delay_angle_0.10_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/bass_delay_angle_0.10_ws_8192.mp3"></audio></div> -->
  <!-- <div class="sound-label">Chord sequence played on a synthesizer.</div>
  <div><audio controls src="./assets/audios/chordula_original.mp3"></audio></div>
  <div><audio controls src="./assets/audios/chordula_angle_0.01_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/chordula_angle_0.01_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/chordula_angle_0.01_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/chordula_angle_0.05_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/chordula_angle_0.05_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/chordula_angle_0.05_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/chordula_angle_0.10_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/chordula_angle_0.10_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/chordula_angle_0.10_ws_8192.mp3"></audio></div> -->
  <!-- <div class="sound-label">Random pitched triangular signals processed with overdrive</div>
  <div><audio controls src="./assets/audios/circuits_original.mp3"></audio></div>
  <div><audio controls src="./assets/audios/circuits_angle_0.01_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/circuits_angle_0.01_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/circuits_angle_0.01_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/circuits_angle_0.05_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/circuits_angle_0.05_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/circuits_angle_0.05_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/circuits_angle_0.10_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/circuits_angle_0.10_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/circuits_angle_0.10_ws_8192.mp3"></audio></div> -->
  <div class="sound-label">Arrangement of glass sounds</div>
  <div><audio controls src="./assets/audios/glasses_original.mp3"></audio></div>
  <div><audio controls src="./assets/audios/glasses_angle_0.01_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/glasses_angle_0.01_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/glasses_angle_0.01_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/glasses_angle_0.05_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/glasses_angle_0.05_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/glasses_angle_0.05_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/glasses_angle_0.10_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/glasses_angle_0.10_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/glasses_angle_0.10_ws_8192.mp3"></audio></div>
  <div class="sound-label">Electric guitar melody processed with ring modulation (1)</div>
  <div><audio controls src="./assets/audios/guitar_rm_1_original.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_1_angle_0.01_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_1_angle_0.01_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_1_angle_0.01_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_1_angle_0.05_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_1_angle_0.05_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_1_angle_0.05_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_1_angle_0.10_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_1_angle_0.10_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_1_angle_0.10_ws_8192.mp3"></audio></div>
  <div class="sound-label">Electric guitar melody processed with ring modulation (2)</div>
  <div><audio controls src="./assets/audios/guitar_rm_2_original.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_2_angle_0.01_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_2_angle_0.01_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_2_angle_0.01_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_2_angle_0.05_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_2_angle_0.05_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_2_angle_0.05_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_2_angle_0.10_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_2_angle_0.10_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/guitar_rm_2_angle_0.10_ws_8192.mp3"></audio></div>
  <div class="sound-label">Random melody played on a piano and processed with ring modulation (1)</div>
  <div><audio controls src="./assets/audios/piano_rm_1_original.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_1_angle_0.01_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_1_angle_0.01_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_1_angle_0.01_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_1_angle_0.05_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_1_angle_0.05_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_1_angle_0.05_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_1_angle_0.10_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_1_angle_0.10_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_1_angle_0.10_ws_8192.mp3"></audio></div>
  <div class="sound-label">Random melody played on a piano and processed with ring modulation (2)</div>
  <div><audio controls src="./assets/audios/piano_rm_2_original.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_2_angle_0.01_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_2_angle_0.01_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_2_angle_0.01_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_2_angle_0.05_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_2_angle_0.05_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_2_angle_0.05_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_2_angle_0.10_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_2_angle_0.10_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/piano_rm_2_angle_0.10_ws_8192.mp3"></audio></div>
  <div class="sound-label">Typing machine recording</div>
  <div><audio controls src="./assets/audios/typing_original.mp3"></audio></div>
  <div><audio controls src="./assets/audios/typing_angle_0.01_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/typing_angle_0.01_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/typing_angle_0.01_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/typing_angle_0.05_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/typing_angle_0.05_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/typing_angle_0.05_ws_8192.mp3"></audio></div>
  <div><audio controls src="./assets/audios/typing_angle_0.10_ws_2048.mp3"></audio></div>
  <div><audio controls src="./assets/audios/typing_angle_0.10_ws_4096.mp3"></audio></div>
  <div><audio controls src="./assets/audios/typing_angle_0.10_ws_8192.mp3"></audio></div>

</div>
