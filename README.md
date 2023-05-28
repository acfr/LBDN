# Direct Parameterization of Lipschitz-Bounded Deep Networks

Summary: This paper ([arxiv](https://arxiv.org/abs/2301.11526)) introduces a new parameterization of deep neural networks (both fully-connected and convolutional) with guaranteed Lipschitz bounds, i.e. limited sensitivity to perturbations. The Lipschitz guarantees are equivalent to the tightest-known bounds based on certification via a semidefinite program (SDP), which does not scale to large models. In contrast to the SDP approach, we provide a **direct parameterization**, i.e. a smooth mapping from $\mathbb R^N$ onto the set of weights of Lipschitz-bounded networks.  This enables training via standard gradient methods, without any computationally intensive projections or barrier terms. The new parameterization can equivalently be thought of as either a new layer type (the **sandwich layer**), or a novel parameterization  of standard feedforward networks with parameter sharing  between neighbouring layers. 

A BibTeX entry for LaTeX users:
```
@misc{ruigang2023direct,
    title={Direct Parameterization of Lipschitz-Bounded Deep Networks}, 
    author={Ruigang Wang and Ian R. Manchester},
    booktitle={International Conference on Machine Learning},
    year={2023},
    organization={PMLR}
}
```

## Experiments
1. Install required packages: einops, auto-attack (https://github.com/fra31/auto-attack)
2. Reprduce the experiments ([pretrained models](https://www.dropbox.com/sh/18i3k4vs4mgiif7/AAAM3HbV8KxuGHHNczlaw0YNa?dl=0)) by running commands below. 

### Lipschtz tightness
* Toy example (Tab.1 & Fig.3): m=[train, eval], g=[1,5,10]
```
python main.py --mode [m] --model Toy --gamma [g] --layer Sandwich --scale small --dataset square_wave --epochs 200 
python main.py --mode [m] --model Toy --gamma [g] --layer Orthogon --scale small --dataset square_wave --epochs 200 
python main.py --mode [m] --model Toy --gamma [g] --layer Aol --scale small --dataset square_wave --epochs 200
python main.py --mode train --model Toy --gamma [g] --layer SLL --scale small --dataset square_wave --epochs 200
```

### Empirical robustness
1. Fig. 4:  
* CIFAR-10 (top row): m=[train, eval], g=[1,10,100]
  
```
python main.py --mode [m] --model KWL --layer Sandwich --scale small --gamma [g] --dataset cifar10 --loss multimargin --epochs 100
python main.py --mode [m] --model KWL --layer Orthogon --scale small --gamma [g] --dataset cifar10 --loss multimargin --epochs 100
python main.py --mode [m] --model KWL --layer Aol --scale small --gamma [g] --dataset cifar10 --loss multimargin --epochs 100
python main.py --mode [m] --model KWL --layer Plain --scale small --dataset cifar10 --loss multimargin --epochs 100
python main.py --mode [m] --model Resnet --layer SLL --scale small --gamma [g] --dataset cifar10 --loss multimargin --epochs 100
```

* CIFAR-100 (bottom row): m=[train, eval], g=[1,2,4]
```
python main.py --mode [m] --model KWL --layer Sandwich --scale large --gamma [g] --dataset cifar100 --loss xent --epochs 100
python main.py --mode [m] --model KWL --layer Orthogon --scale large --gamma [g] --dataset cifar100 --loss xent --epochs 100
python main.py --mode [m] --model KWL --layer Aol --scale large --gamma [g] --dataset cifar100 --loss xent --epochs 100
python main.py --mode [m] --model KWL --layer Plain --scale large --dataset cifar100 --loss xent --epochs 100
python main.py --mode [m] --model Resnet --layer SLL --scale large --gamma [g] --dataset cifar100 --loss xent --epochs 100
```

2. Fig. 5:
* CIFAR-10 (left): s=[123,43,13,7,365]
```
python main.py --mode train --model KWL --layer Sandwich --scale small --gamma 100 --dataset cifar10 --loss multimargin --epochs 100 --seed [s]
python main.py --mode train --model KWL --layer Orthogon --scale small --gamma 100 --dataset cifar10 --loss multimargin --epochs 100 --seed [s]
python main.py --mode train --model KWL --layer Aol --scale small --gamma 100 --dataset cifar10 --loss multimargin --epochs 100 --seed [s]
python main.py --mode train --model KWL --layer Plain --scale small --dataset cifar10 --loss multimargin --epochs 100 --seed [s]
python main.py --mode train --model Resnet --layer SLL --scale small --gamma 100 --dataset cifar10 --loss multimargin --epochs 100 --seed [s]
```
* CIFAR-100 (right): s=[123,43,13,7,365]
```
python main.py --mode train --model KWL --layer Sandwich --scale large --gamma 10 --dataset cifar100 --loss xent --epochs 100 --seed [s]
python main.py --mode train --model KWL --layer Orthogon --scale large --gamma 10 --dataset cifar100 --loss xent --epochs 100 --seed [s]
python main.py --mode train --model KWL --layer Aol --scale large --gamma 10 --dataset cifar100 --loss xent --epochs 100 --seed [s]
python main.py --mode train --model KWL --layer Plain --scale large --dataset cifar100 --loss xent --epochs 100 --seed [s]
python main.py --mode train --model Resnet --layer SLL --scale large --gamma 10 --dataset cifar100 --loss xent --epochs 100 --seed [s]
```

3. Tab. 2:
* CIFAR-100: m=[train, eval], c=[small,medium,large], s=[123,43,13]
```
python main.py --mode [m] --model KWL --layer Sandwich --scale [c] --gamma 2 --dataset cifar100 --loss xent --epochs 100 --normalized --seed [s]
python main.py --mode [m] --model KWL --layer Orthogon --scale [c] --gamma 2 --dataset cifar100 --loss xent --epochs 100 --normalized --seed [s]
python main.py --mode [m] --model KWL --layer Aol --scale [c] --gamma 2 --dataset cifar100 --loss xent --epochs 100 --normalized --seed [s]
python main.py --mode [m] --model Resnet --layer SLL --scale [c] --gamma 2 --dataset cifar100 --loss xent --epochs 100 --normalized --seed [s]
```

* Tiny-Imagenet: m=[train, eval], c=[small,medium,large], s=[123,43,13]
```
python main.py --mode [m] --model KWL --layer Sandwich --scale [c] --gamma 2 --dataset tiny_imagenet --loss xent --epochs 100 --normalized --seed [s]
python main.py --mode [m] --model KWL --layer Orthogon --scale [c] --gamma 2 --dataset tiny_imagenet --loss xent --epochs 100 --normalized --seed [s]
python main.py --mode [m] --model KWL --layer Aol --scale [c] --gamma 2 --dataset tiny_imagenet --loss xent --epochs 100 --normalized --seed [s]
python main.py --mode [m] --model Resnet --layer SLL --scale [c] --gamma 2 --dataset tiny_imagenet --loss xent --epochs 100 --normalized --seed [s]
```

### Certified robustness
1. Tab. 3 & 4: 
* CIFAR-100: m=[train, eval], s=[123,43,13]
```
python main.py --mode [m] --model DNN --layer Sandwich --scale small --gamma 1 --dataset cifar100 --loss xent --epochs 400 --cert_acc --seed [s]
```

* Tiny-Imagenet: m=[train, eval], s=[123,43,13]
```
python main.py --mode [m] --model DNN --layer Sandwich --scale small --gamma 1 --dataset tiny_imagenet --loss xent --epochs 400 --cert_acc --seed [s]
```

