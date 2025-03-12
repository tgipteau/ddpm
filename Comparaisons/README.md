<h2>Comparaisons</h2>

On s'intéresse ici à la comparaison de DDPM avec un VAE et un GAN, ces trois modèles ayant été codés à la main.

- <i>samples.png</i> : images MNIST générées par le DDPM à l'epoch 82

  Les deux fichier suivants sont conçus sur le même modèle : on construit les réseaux dans un premier temps, puis on met à disposition une fonction train et une fonction generate. Ce sont les deux seules fonctions à appeler à priori.
  Les poids enregistrés et les images générées ne sont pas sauvegardées sur GitHub.
  
- <i>VAE_generator.py<i> : code pour le générateur VAE
- <i>GAN_generator.py</i> : code pour le générateur GAN

- <i>comparateur.ipynb</i> : notebook contenant les travaux de comparaison et d'évaluation, utilisant les générations des trois modèles. Il est nécéssaire d'avoir, dans le même dossier que ce code, trois dossiers, respectivement "GAN_generated", "VAE_generated" et "DDPM_generated", contenant 100 images générées par chaque modèle.
  Pour DDPM generated, on se réferrera au dossier DDPM, contenant les codes nécessaires.

- <i>distribution.png</i> : Distributions des générations des trois modèles, figure issue du notebook.
- <i> best </i> : Dossier contenant les meilleurs samples pour chaque digit de chaque modèle, figures issues du notebook également.
  
