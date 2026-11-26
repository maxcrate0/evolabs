# EvoLab

Visualizador desktop para treinar populações de redes neurais por seleção, elitismo e mutação gaussiana. Inclui Flappy Bird e o Dinossauro do Google como desafios integrados.

## Executar

Use Python 3.10 ou superior, com Tkinter para a janela (incluído na instalação padrão do Python para Windows).

```powershell
python -m pip install -r requirements.txt
python main.py
```

Em **Novo treino**, escolha o desafio e a rede dos agentes. O **Flappy Bird** abre com uma prévia e está pronto para treinar. Para usar um ambiente próprio, escolha **Importar desafio**. Abra **Ajustes avançados** para mudar o número de agentes e de rodadas, as elites preservadas, o grupo de pais, a taxa de mutação por peso, o desvio padrão do ruído gaussiano e a semente.

Acima da simulação, **Melhor agente** exibe a rede com a maior pontuação da rodada e **Todos os agentes** exibe a população. A escolha vale para a próxima avaliação.

Abaixo da simulação, os botões de **Velocidade** podem ser usados durante o treino.

- **Normal** é a opção inicial. Nos desafios integrados, a reprodução dura três vezes mais que em **Rápido**, com movimento interpolado entre os quadros. Quadros sem identificadores de movimento usam uma redução mais leve de velocidade.
- **Rápido** usa os tempos de exibição definidos pelo ambiente, como nas versões anteriores.
- **Headless** encerra a animação em curso. A partir da próxima avaliação, o ambiente calcula as pontuações sem gerar quadros. A janela continua exibindo as métricas.

Na reprodução de uma rede salva, use **Normal** ou **Rápido**. **Parar treino** encerra a execução depois da chamada ao ambiente em andamento. Passe o cursor sobre o gráfico para conferir a melhor pontuação e a média de uma rodada. Cada geração chama a avaliação exatamente uma vez. Os modos visuais reproduzem os quadros retornados por essa chamada.

Para uma execução sem janela, use o terminal.

```powershell
python main.py --headless --generations 100 --template multicamadas
python main.py --headless --help
```

## Desafios integrados

Na janela, escolha um dos exemplos em **Novo treino → Desafio** e clique em **Começar treino**. Todos usam a mesma interface de visualização e salvamento.

| Desafio | O que a rede aprende | Como pontua | Nome no terminal |
| --- | --- | --- | --- |
| **Flappy Bird** | Bater as asas no momento certo para atravessar os canos. | Ganha um ponto por passo de simulação que sobrevive. Bater nos canos, no chão ou no teto encerra o voo. | `flappy-bird` |
| **Dinossauro do Google** | Saltar sobre os cactos enquanto a corrida acelera. | Ganha um ponto por passo de simulação que sobrevive. A corrida termina ao atingir um cacto. | `chrome-dino` |

Para rodar 20 gerações de cada desafio sem abrir uma janela, use `--challenge`. Sem essa opção, o EvoLab usa **Flappy Bird**.

```powershell
python main.py --headless --challenge flappy-bird --generations 20
python main.py --headless --challenge chrome-dino --generations 20
```

Os dois desafios geram o percurso a partir da semente da rodada. A implementação está em [challenges.py](challenges.py). Treinos salvos dos desafios anteriores ainda podem ser reproduzidos e continuados.

## Salvar, assistir e continuar

O EvoLab salva um arquivo `.evolab` em `treinos/` ao concluir o treino. Ao usar **Parar treino**, ele salva as rodadas já avaliadas. **Abrir treino…** carrega o histórico e disponibiliza **Assistir à melhor rede** e **Continuar treino…**. A reprodução coloca a melhor rede da última população avaliada para jogar sozinha, com a semente daquela rodada. A continuação usa a população inteira, as pontuações e os estados dos geradores aleatórios para criar a próxima geração. Cada continuação salva um novo arquivo com o histórico acumulado.

O formato `.evolab` é um ZIP com `history.csv` (log por rodada), `best.npy` (pesos da melhor rede da última rodada), `population.npy` e `scores.npy` (população e pontuações finais) e `training.json` (configuração, caminho dos plugins e estados aleatórios). Arquivos Python e executáveis externos precisam continuar disponíveis nos caminhos registrados, com suas respectivas dependências. Abra e execute treinos salvos de fontes em que você confia.

No terminal, `--generations` acrescenta rodadas quando combinado com `--resume`. Use `--output-dir` para escolher outro diretório de saída.

```powershell
python main.py --headless --resume treinos/treino-exemplo.evolab --generations 25
```

## Arquiteturas

Em **Rede dos agentes**, escolha **Equilibrada**, **Mais completa**, **Mais simples** ou selecione **JSON personalizado** para editar as camadas. As dimensões de entrada e saída vêm do ambiente.

```json
{"hidden": [32, 16], "activation": "relu", "output": "tanh"}
```

`hidden` aceita uma lista de larguras positivas. Uma lista vazia cria uma rede com uma única camada. As ativações disponíveis são `linear`, `tanh`, `relu`, `sigmoid` e `softmax`. A CLI também aceita esse JSON em arquivo com `--architecture rede.json`.

Em **Arquivo Python** ou com `--architecture rede.py`, o arquivo deve exportar `build(inputs, outputs)`. A função retorna um objeto com os seguintes atributos.

| Atributo | Contrato |
| --- | --- |
| `genome_size` | Inteiro positivo com a quantidade de parâmetros por indivíduo. |
| `initialize(rng, count)` | Recebe um `numpy.random.Generator` e retorna um array de pesos finitos `(count, genome_size)`. |
| `forward(weights, observations)` | Recebe arrays `float32` de formas `(N, genome_size)` e `(N, inputs)`. Retorna ações de forma `(N, outputs)`. Cada linha de observações usa os pesos da linha correspondente. |

A rede pode usar NumPy ou JAX. O exemplo com JAX usa `jax.vmap` e `jax.jit` para inferência em lote. Instale JAX separadamente para rodá-lo.

```powershell
python -m pip install jax
python main.py --headless --architecture examples/network_jax.py
```

## Ambientes Python

O ambiente exporta duas funções.

```python
def describe():
    return {"inputs": 4, "outputs": 2}

def evaluate(predict, weights, *, view, seed):
    # Simule os indivíduos e calcule suas pontuações.
    # predict(weights, observations) retorna as ações do lote.
    return {"scores": scores, "frames": frames}
```

`weights` é a matriz somente leitura da população inteira. Chame `predict` por passo com um lote, ou use subconjuntos com pesos e observações alinhados. A `seed` é um inteiro gerado a cada avaliação a partir da semente do experimento.

`scores` contém uma pontuação finita por indivíduo na ordem da entrada. Quanto maior, melhor. Use `view="none"` para pular toda a captura e omitir `frames`. Nos modos `best` e `population`, devolva os quadros da trajetória correspondente. Em empate, use o primeiro indivíduo de maior pontuação para que a visualização e o elitismo sigam a mesma ordem. Consulte `Scroller.evaluate` em [challenges.py](challenges.py) para um exemplo de inferência de toda a população em lote.

## Ambientes executáveis

Em **Desafio**, escolha **Importar desafio** e selecione o executável. O visualizador mantém um processo aberto durante todo o treino. O protocolo usa um objeto JSON UTF-8 por linha, em `stdin` e `stdout`. Faça flush a cada resposta e escreva logs em `stderr`.

| Direção | Exemplo |
| --- | --- |
| EvoLab → ambiente | `{"type":"describe","version":1}` |
| Ambiente → EvoLab | `{"type":"description","inputs":1,"outputs":1}` |
| EvoLab → ambiente | `{"type":"evaluate","view":"best","seed":7,"network":{"genome_size":2,"architecture":null},"weights":[[0.25,0],[0.75,0]]}` |
| Ambiente → EvoLab | `{"type":"predict","observations":[[0.5],[0.5]]}` |
| EvoLab → ambiente | `{"type":"prediction","actions":[[0.125],[0.375]]}` |
| Ambiente → EvoLab | `{"type":"result","scores":[-0.375,-0.125],"frames":[]}` |

O ambiente pode pedir inferência quantas vezes precisar antes de enviar `result`. Use o campo opcional `indices` para escolher as linhas dos pesos na ordem das observações. Para diminuir o custo de comunicação, envie observações em lote. Os templates densos também enviam `network.architecture` com `type: "dense"`, `inputs`, `outputs`, `hidden`, `activation` e `output`. Um executável que implemente essa arquitetura pode calcular as ações localmente. Cada camada usa os pesos da matriz em ordem de linhas, seguidos pelos vieses. Redes Python fornecem `architecture: null` e podem usar a inferência pelo protocolo.

Em caso de erro, responda com `{"type":"error","message":"descrição"}`. O exemplo [environment_stdio.py](examples/environment_stdio.py) pode ser executado pelo protocolo via interpretador para testar a integração.

```powershell
python main.py --headless --external --environment examples/environment_stdio.py
```

## Formato dos quadros

Ambientes Python e executáveis usam a mesma estrutura. Um quadro tem largura e altura lógicas, fundo opcional, duração em milissegundos e uma lista de formas na ordem em que devem ser desenhadas. A janela mantém a proporção do quadro ao redimensionar.

Para suavizar o movimento no modo **Normal**, use o campo opcional `id` (texto ou inteiro) em círculos e retângulos que se movem. Mantenha um identificador único por quadro e use o mesmo valor para o mesmo objeto em quadros consecutivos. A interface interpola a posição; mudanças de tamanho ou cor são aplicadas no quadro seguinte.

```json
{
  "width": 640,
  "height": 384,
  "duration_ms": 32,
  "background": "#141d2d",
  "shapes": [
    {"kind": "circle", "id": "agente", "x": 32, "y": 40, "r": 5, "fill": "#71f0c2"},
    {"kind": "rect", "x": 90, "y": 20, "w": 16, "h": 80, "outline": "#52617d"},
    {"kind": "line", "points": [32, 40, 60, 75, 95, 120], "width": 2, "fill": "#81a0d3"}
  ]
}
```
