---
layout: default
title: Fractional Fourier Sound Synthesis
---

<div style="position: relative; width: 100%; height: 60px; margin-bottom: 20px;">
  <img src="assets/img/uc.svg" alt="UC Logo"
       style="position: absolute; top: 50%; left: 0; max-height: 60px; max-width: 50%; height: auto; width: auto; transform: translateY(-50%);">
  <img src="assets/img/upf.png" alt="UPF Logo"
       style="position: absolute; top: 50%; right: 0; max-height: 60px; max-width: 50%; height: auto; width: auto; transform: translateY(-50%);">
</div>

<div style="text-align: center">
<h1>Fractional Fourier Sound Synthesis</h1>
<p>
  <a href="https://cordutie.github.io/"><strong>Esteban Gutiérrez</strong></a><sup>1</sup>, 
  <a href="https://rodrigocadiz.com/"><strong>Rodrigo Cádiz</strong></a><sup>2,3</sup>, 
  <a href="https://www.ing.uc.cl/academicos-e-investigadores/carlos-alberto-sing-long-collao/"><strong>Carlos Sing Long</strong></a><sup>4</sup>, 
  <a href="https://ffont.github.io/"><strong>Frederic Font</strong></a><sup>1</sup>, and
  <a href="https://www.upf.edu/web/xavier-serra"><strong>Xavier Serra</strong></a><sup>1</sup>
</p>

<p>
  <sup>1</sup> <em>Department of Information and Communications Technologies, Universitat Pompeu Fabra</em><br>
  <sup>2</sup> <em>Music Institute, Pontificia Universidad Católica de Chile</em><br>
  <sup>3</sup> <em>Department of Electrical Engineering, Pontificia Universidad Católica de Chile</em><br>
  <sup>4</sup> <em>Instituto de Ingeniería Matemática y Computacional, Pontificia Universidad Católica de Chile</em>
</p>
</div>

<div style="text-align: center; margin-top: 1em; margin-bottom: -0.3em;">
  <a href="https://www.arxiv.org/abs/2506.09189" 
    style="display: inline-block; background: #3498db; color: white; padding: 0.6em 1em; border-radius: 5px; text-decoration: none; margin: 0.3em 0em;">
    📄 Paper
  </a>
  <a href="https://github.com/cordutie/frft_sound_synthesis" 
    style="display: inline-block; background: #3498db; color: white; padding: 0.6em 1em; border-radius: 5px; text-decoration: none; margin: 0.3em 0em;">
    <img src="/assets/img/gh_logo.png" alt="GitHub" width="15" style="filter: invert(1); vertical-align: -0.2em;" /> Experiments repository
  </a>
  <a href="#" onclick="event.preventDefault(); navigator.clipboard.writeText('@inproceedings{gutierrez2025frft,\n title     = {Fractional Fourier Sound Synthesis},\n author    = {Esteban Gutiérrez and Rodrigo Cádiz and Carlos Sing Long and Frederic Font and Xavier Serra},\n booktitle = {Proceedings of the 50th International Computer Music Conference (ICMC) 2025},\n year      = {2025},\n address   = {Boston, USA},\n month     = {June},\n note      = {8--14 September 2025}\n}'); alert('Copied to clipboard!');"
    style="display: inline-block; background: #3498db; color: white; padding: 0.6em 1em; border-radius: 5px; text-decoration: none; margin: 0.3em 0em;">
    📚 Bibtex
  </a>
</div>

<div style="margin-top: 20px;"></div>
<p>
This webpage provides supplementary materials for our paper <em>"Fractional Fourier Sound Synthesis"</em>, to be presented at the International Computer Music Conference (ICMC) 2025 in Boston, USA.
</p>

<div style="margin-top: 20px;"></div>
<h2><strong>1. Introduction</strong></h2>
{% include_relative 1_introduction.md %}

<div style="margin-top: 40px;"></div>
<h2><strong>2. Methods</strong></h2>
{% include_relative 2_methods.md %}

<div style="margin-top: 40px;"></div>
<h2><strong>3. Experiments</strong></h2>
{% include_relative 3_experiments.md %}

<div style="margin-top: 40px;"></div>
<div style="display: flex; justify-content: center;">
  <div style="border-left: 4px solid rgb(200, 200, 200); background:rgb(230, 230, 230); padding: 1em 1.2em; margin: 1.5em 0; border-radius: 8px; max-width: 300px; width: 100%;">
    <strong>Legend:</strong>
    <ul style="list-style: none; padding: 0.5em 0 0 0; margin: 0;">
      <li>🎧 Sound examples included</li>
      <li>📊 Numerical experiments included</li>
      <li>📖 Theory included</li>
      <li>🚧 Still under construction</li>
    </ul>
  </div>
</div>

<div style="margin-top: 40px;"></div>
<h2><strong>Acknowledgements</strong></h2>

This research was supported by ANID Fondecyt Regular Grant #1230926, ANID Anillo ATE220041, Government of Chile, and the project "IA y Música: Cátedra en Inteligencia Artificial y Música (TSI-100929-2023-1)" funded by the "Secretaría de Estado de Digitalización e Inteligencia Artificial and the Unión Europea-Next Generation EU". We would also like to thank Diego Vera for his contributions to an early version of this project during his undergraduate research.

<div style="justify-content: center; width: 100%;">
  <img src="assets/img/chair.png" alt="Funding" style="top: 50%; right: 0; width: 100%;">
</div>