# Simulador de Trajetória Tau Rocket Team
### Desenvolvido por:
* Andres Benoit (Engenharia Aeroespacial - UFSM) - andres.benoit@acad.ufsm.br

### Visão geral:
O simulador resolve o trajeto de ascensão de um veículo lançador de propulsão híbrida em 6DOF, e em 3DOF a parte de descida e recuperação do veículo. Além disso, o simulador implementa uma amostragem de perfil de vento com base em dados fornecidos pelo NOAA com base em funções gaussianas. O algoritmo foi desenvolvido com o objetivo de analisar a trajetória do foguete Photon, desenvolvido pela equipe de foguete modelismo Tau Rocket Team.

![alt text](https://github.com/Andres2704/RocketTrajectorySimulation/blob/master/images/logo_horizontal_tau.png)

### Entradas:
Para utilizar o simulador deve-se ter os seguintes dados:
* Condições Iniciais
* A geometria do veículo
* Dados de recuperação
* Dados de simulação
* Dados propulsivos e estruturas

Todos essas informações são alteradas no arquivo main.py

### Saídas:
* Trajetória de ascensão em 6DOF e descida 3DOF
* Resposta dinâmica do veículo (Estabilidade dinâmica)
* Resposta estática do veículo (Estabilidade estática)
* Ângulos aerodinâmicos
