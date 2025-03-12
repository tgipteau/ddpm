<h2>DDPM</h2>

On trouve dans ce dossier les codes relatifs à l'implémentation de DDPM sur MNIST, ainsi qu'un UNET fonctionnel pour celui-ci. 

- <i> DDPM_generated </i> : samples de génération de DDPM. Pour appliquer les codes de "Comparaisons", il sera nécessaire de générer 100 images, plutôt que les 10 présentées ici à titre d'exemple. On copiera ensuite ce dossier dans le dossier Comparaisons, pour l'utiliser avec comparateur.ipynb.
- <i> config_ddpm.yaml </i> : fichier de configuration du DDPM, lu par main.py et unet_mnist.py. C'est à travers ce fichier que l'on doit ajuster le comportement du DDPM dans un premier temps.
- <i> unet_mnist.py </i> : modèle unet adapté au format des images mnist. Utilisé par main.py si unet_model = "unet_mnist" dans le yaml.
- <i> unet_cifar.py </i> : résidu d'un essai d'application de DDPM aux images Cifar, par concluant. Ne pas utiliser ce modèle.

- <i> main.py </i> : code principal du ddpm. On décrit l'influence du .yaml ci-dessous :

  -- on pensera à renseigner le bon device dans le yaml. Sur mac, on a utilisé "mps". Les autres choix possibles sont "cuda" et "cpu".

  -- si old_model=True, train=False : on utilise le modèle enregistré, dont le nom est renseigné dans "store_path". C'est le fonctionnement par défaut. Dans ce cadre, il n'y a pas de routine d'entraînement et le modèle renseigné est chargé. Notez qu'il faut alors ne pas changer les paramètres du DDPM.
    Dans ce cas, la seule fonction appelée est generate_for_comparison, qui génère par défaut 100 images dans le dossier DDPM_generated (à créer en amont).

  -- si old_model=True, train=True : on utilise l'ancien modèle qu'on continue à entraîner pour le nombre d'epochs renseigné. Penser alors à modifier store_path pour ne pas perdre le modèle précédent. Ce paramétrage permet un entraînement sur plusieurs sessions.

  -- si old_model=False, train=True : on repart de zéro ; routine d'apprentissage à partir d'un modèle initialisé aléatoirement.
  
  -- les résultats présentés sont issus de la graine aléatoire 0, qui donnera toujours les mêmes sorties. Pour obtenir des résultats nouveaux, il est nécessaire de changer cette valeur.

  
  
  
