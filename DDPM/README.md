<h2>DDPM</h2>

On trouve dans ce dossier les codes relatifs à l'implémentation de DDPM sur MNIST, ainsi qu'un UNET fonctionnel pour celui-ci. 

- <i> DDPM_generated </i> : samples de génération de DDPM. Pour appliquer les codes de "Comparaisons", il sera nécessaire de générer 100 images, plutôt que les 10 présentées ici à titre d'exemple.
- <i> config_ddpm.yaml </i> : fichier de configuration du DDPM, lu par main.py et unet_mnist.py. C'est à travers ce fichier que l'on doit ajuster le comportement du DDPM dans un premier temps.
- <i> unet_mnist.py </i> : modèle unet adapté au format des images mnist. Utilisé par main.py si unet_model = "unet_mnist" dans le yaml.
- <i> unet_cifar.py </i> : résidu d'un essai d'application de DDPM aux images Cifar, par concluant. Ne pas utiliser ce modèle.

- <i> main.py </i> : code principal du ddpm ; son contenu est décrit ci-dessous

<h3> main.py </h3>

lignes 1 à 65 : imports et chargement de la configuration inscrite dans config_ddpm.yaml.
ligne 69 : déclaration de la classe "MyDDPM"
ligne 
