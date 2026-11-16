import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class AnalisadorDeDados:
    """Carrega, limpa e analisa a base de passageiros do Titanic.

    Toda a lógica de dados (Pandas) e de gráficos (Matplotlib/Seaborn)
    fica encapsulada aqui; o main.py apenas chama os métodos.
    """

    def __init__(self, caminho_arquivo):
        self.df = None
        try:
            # Leitura da base com Pandas
            self.df = pd.read_csv(caminho_arquivo)

            # ---------------- Limpeza dos dados ----------------
            # 'Cabin' tem ~77% de nulos: remover a coluna evita que o
            # dropna() final apague quase toda a base.
            self.df = self.df.drop(columns=["Cabin"], errors="ignore")

            # fillna(): preenche nulos com mediana (numéricos) e moda (texto)
            self.df["Age"] = self.df["Age"].fillna(self.df["Age"].median())
            self.df["Fare"] = self.df["Fare"].fillna(self.df["Fare"].median())
            self.df["Embarked"] = self.df["Embarked"].fillna(
                self.df["Embarked"].mode()[0]
            )

            # dropna(): garante que não sobrou nenhuma linha com nulo
            self.df = self.df.dropna()

            # Coluna auxiliar legível para os gráficos
            self.df["Resultado"] = self.df["Survived"].map(
                {0: "Não sobreviveu", 1: "Sobreviveu"}
            )

            print("\nBase de dados carregada com sucesso!")
            print(f"Registros disponíveis: {len(self.df)}")
        except FileNotFoundError:
            self.df = None
            print(f"Erro: arquivo '{caminho_arquivo}' não encontrado.")
        except Exception as erro:
            self.df = None
            print(f"Erro ao carregar os dados: {erro}")

    # ------------------------------------------------------------------
    # Estatísticas
    # ------------------------------------------------------------------
    def mostrar_estatisticas(self):
        if self.df is None:
            return

        total = len(self.df)
        sobreviventes = int(self.df["Survived"].sum())

        print("\n===== ESTATÍSTICAS GERAIS =====")
        print(f"Total de passageiros : {total}")
        print(f"Sobreviventes        : {sobreviventes} "
              f"({sobreviventes / total * 100:.1f}%)")
        print(f"Média de idade       : {self.df['Age'].mean():.2f} anos")
        print(f"Média das tarifas    : {self.df['Fare'].mean():.2f}")

        print("\nTaxa de sobrevivência por sexo (%):")
        print((self.df.groupby("Sex")["Survived"].mean() * 100).round(1))

        print("\nTarifa média por classe:")
        print(self.df.groupby("Pclass")["Fare"].mean().round(2))

    # ------------------------------------------------------------------
    # Gráficos
    # ------------------------------------------------------------------
    def sobreviventes_por_sexo(self):
        """Gráfico de barras: sobreviventes por sexo (groupby)."""
        if self.df is None:
            return

        dados = self.df.groupby("Sex")["Survived"].sum()

        plt.figure(figsize=(8, 5))
        dados.plot(kind="bar", color=["salmon", "steelblue"])

        plt.title("Quantidade de Sobreviventes por Sexo")
        plt.xlabel("Sexo")
        plt.ylabel("Quantidade de sobreviventes")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def distribuicao_idades(self):
        """Histograma: idades dos passageiros (filtro booleano)."""
        if self.df is None:
            return

        # Filtro booleano: considera apenas idades válidas (> 0)
        dados = self.df[self.df["Age"] > 0]

        plt.figure(figsize=(8, 5))
        sns.histplot(data=dados, x="Age", bins=20, color="royalblue", kde=True)

        plt.title("Distribuição das Idades dos Passageiros")
        plt.xlabel("Idade (anos)")
        plt.ylabel("Quantidade de passageiros")
        plt.tight_layout()
        plt.show()

    def idade_e_tarifa(self):
        """Dispersão: idade x tarifa (filtro booleano remove outliers)."""
        if self.df is None:
            return

        # Filtro booleano: tarifas acima de 300 distorcem o gráfico
        dados = self.df[self.df["Fare"] < 300]

        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=dados, x="Age", y="Fare", hue="Resultado", palette="Set1"
        )

        plt.title("Relação entre Idade e Tarifa (tarifas < 300)")
        plt.xlabel("Idade (anos)")
        plt.ylabel("Tarifa paga")
        plt.legend(title="Resultado")
        plt.tight_layout()
        plt.show()

    def sobreviventes_por_classe(self):
        """Gráfico de linha: % de sobrevivência por classe (groupby)."""
        if self.df is None:
            return

        dados = self.df.groupby("Pclass")["Survived"].mean() * 100

        plt.figure(figsize=(8, 5))
        plt.plot(dados.index, dados.values, marker="o", color="green")

        plt.title("Percentual de Sobreviventes por Classe")
        plt.xlabel("Classe do passageiro")
        plt.ylabel("Sobreviventes (%)")
        plt.xticks([1, 2, 3])
        plt.tight_layout()
        plt.show()
