
<p>
This paper explores the innovative application of the Fractional Fourier Transform (FrFT) in sound synthesis, highlighting its potential to redefine time-frequency analysis in audio processing. As an extension of the classical Fourier Transform, the FrFT introduces fractional order parameters, enabling a continuous interpolation between time and frequency domains and unlocking unprecedented flexibility in signal manipulation. Crucially, the FrFT also opens the possibility of directly synthesizing sounds in the \(\alpha\)-domain, providing a unique framework for creating timbral and dynamic characteristics unattainable through conventional methods. This work delves into the mathematical principles of the FrFT, its historical evolution, and its capabilities for synthesizing complex audio textures. Through experimental analyses, we showcase novel sound design techniques, such as \(\alpha\)-synthesis and \(\alpha\)-filtering, which leverage the FrFT’s time-frequency rotation properties to produce innovative sonic results. The findings affirm the FrFT’s value as a transformative tool for composers, sound designers, and researchers seeking to push the boundaries of auditory creativity.
</p>

<!-- <div style="overflow-x: auto; max-width: 80%; margin: 0 auto; padding: 10px; box-sizing: border-box;">
  <div style="text-align: center; margin: 30px 0;">
    <img src="./assets/img/architecture.svg" alt="TexDSP architecture" style="max-width: 100%; height: auto;">
  </div>
  <p style="text-align: center; font-size: 0.85em; color: #666;"><strong>Figure 1.1.</strong> <code>TexDSP</code> architecture diagram. </p>  
</div> -->

<!-- 
<div style="overflow-x: auto; max-width: 80%; margin: 0 auto; padding: 10px; box-sizing: border-box;">
  <div style="display: grid; grid-template-columns: repeat(3, minmax(200px, 1fr)); gap: 20px; text-align: center;">

  <div style="font-weight: bold;"><strong>Fire Model</strong></div>
  <div style="font-weight: bold;"><strong>Water Model</strong></div>
  <div style="font-weight: bold;"><strong>Wind Model</strong></div>

  <div><img src="./assets/img/fire.gif" alt="Fire Model" style="max-width: 100%;" /></div>
  <div><img src="./assets/img/bubbles_2.gif" alt="Bubbles Model" style="max-width: 100%;" /></div>
  <div><img src="./assets/img/wind.gif" alt="Wind Model" style="max-width: 100%;" /></div>

  <div>
    <audio controls style="width: 100%;">
      <source src="./assets/audios/texdsp_timbre_transfer/fire_to_fire.mp3" type="audio/mpeg" />
      Your browser does not support the audio element.
    </audio>
  </div>
  <div>
    <audio controls style="width: 100%;">
      <source src="./assets/audios/texdsp_timbre_transfer/bubbles_to_bubbles.mp3" type="audio/mpeg" />
      Your browser does not support the audio element.
    </audio>
  </div>
  <div>
    <audio controls style="width: 100%;">
      <source src="./assets/audios/texdsp_timbre_transfer/wind_to_wind.mp3" type="audio/mpeg" />
      Your browser does not support the audio element.
    </audio>
  </div>
</div>
</div>

<div style="overflow-x: auto; max-width: 80%; margin: 0 auto; padding: 10px; box-sizing: border-box; text-align: center; font-size: 0.85em; ">
<strong>Sound Examples 1.1.</strong> Sound examples generated using the <code>TexDSP</code> architecture trained using the <code>TexStat</code> loss function.</div> -->