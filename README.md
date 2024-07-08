# Classificação de Preços de Reserva em Hotéis com ML

## 🌐 Sobre o Projeto!

Este projeto consiste na criação de um serviço de machine learning para classificação de preços de reserva em hotéis, utilizando o Hotel Reservations Dataset. O modelo foi treinado com o SageMaker da AWS, rodando localmente e a inferência foi feita por meio de uma API desenvolvida em Python.

---

## 🎁 Funcionalidades

- Este projeto oferece diversas funcionalidades essenciais para a classificação de reservas de hotel com base em faixas de preço por quarto, utilizando um modelo de machine learning treinado no AWS SageMaker. A seguir, apresentarei as principais funcionalidades:

**1. Carregamento e Preparação dos Dados**

- Notebooks Jupyter: Utilizei notebooks para carregar, explorar e preparar os dados, incluindo limpeza, criação de novas features e armazenamento dos dados processados no AWS S3 e AWS RDS.
- Interação com RDS: Conectei ao banco de dados relacional (RDS - MySQL) da AWS para executar consultas SQL e manipular os dados diretamente.

**2. Treinamento do Modelo**

- AWS SageMaker: Usei o AWS SageMaker para treinar um modelo de machine learning, utilizando dados armazenados no S3 e configurando o treinamento nos notebooks.
- Modelo Random Forest: Optado pelo algoritmo Random Forest devido à sua robustez e alta performance em tarefas de classificação.

**3. Desenvolvimento da API**

**FastAPI**

- FastAPI: Foi desenvolvida uma API com o framework FastAPI, que oferece uma interface RESTful para predições, configurada para carregar o modelo treinado a partir do S3.

**4. Containerização**

- Docker: Usado Docker para containerizar a aplicação, garantindo um ambiente de execução consistente em diferentes máquinas.

**5. Deploy na AWS**

- EC2: A aplicação pode ser implantada na AWS usando Amazon ECS, EKS ou instâncias EC2. A utilização de containers Docker facilita o deploy e a escalabilidade da aplicação.
- AWS S3: O modelo treinado e os dados são armazenados no Amazon S3, permitindo fácil acesso e gerenciamento.

**Endpoint**

- /api/v1/predict: Endpoint POST que recebe um JSON com os dados da reserva e retorna a classificação (faixa de preço).

---

# Construção do Modelo

O [Hotel Reservations Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) contém informações sobre reservas em hotéis e será utilizado para classificar os dados por faixa de preços. A equipe criou uma nova coluna chamada `label_avg_price_per_room` para categorizar as reservas em três faixas de preço:

1. `1` para `avg_price_per_room` ≤ 85
2. `2` para 85 < `avg_price_per_room` < 115
3. `3` para `avg_price_per_room` ≥ 115

O dataset original e o processado foram armazenados no AWS RDS e o modelo treinado foi salvo no S3.

---

## 📂 Estrutura do Repositório

    ├── assets/                          # Diretório para armazenar ativos como imagens usadas no README
    │   └── sprint4-5.jpg                # Imagem usada no README
    ├── src/                             # Diretório que armazena o código-fonte do projeto
    │   ├── api/                         # Diretório para o código do serviço de inferência
    │   │   ├── app/                     # Subdiretório contendo os principais componentes da aplicação
    │   │   │   ├── main.py              # Ponto de entrada da aplicação FastAPI
    │   │   │   ├── controllers.py       # Arquivo contendo a lógica de controle da aplicação
    │   │   │   ├── models.py            # Arquivo contendo o carregamento e gerenciamento do modelo
    │   │   │   └── views.py             # Arquivo contendo as rotas/endpoints da aplicação
    │   │   ├── docker-compose.yml       # Arquivo de configuração do Docker Compose para orquestração de contêineres
    │   │   ├── Dockerfile               # Arquivo para definição da imagem Docker da aplicação
    │   │   ├── requirements.txt         # Lista de dependências Python necessárias para o serviço de inferência
    │   ├── python/                      # Diretório para scripts Python auxiliares e notebooks
    │   │   ├── sagemaker/               # Subdiretório para scripts relacionados ao Amazon SageMaker
    │   │   │   ├── Treinamento.ipynb    # Notebook para treinamento do modelo no SageMaker
    │   │   │   └── requirements.txt     # Lista de dependências Python necessárias para o treinamento no SageMaker
    │   │   ├── scripts/                 # Subdiretório para scripts de manipulação de dados
    │   │   │   ├── csv_to_rds.ipynb     # Notebook para converter CSV para RDS
    │   │   │   ├── rds_to_csv.ipynb     # Notebook para converter RDS para CSV
    │   │   │   └── requirements.txt     # Lista de dependências Python necessárias para os scripts de manipulação de dados
    ├── .gitignore                       # Arquivo para especificar quais arquivos/diretórios o Git deve ignorar
    └── README.md                        # Documentação do projeto

## 🔧 Pré-requisitos

`Python 3.11`, `AWS CLI`, `Jupyter Notebook` e `Docker`

---

## 🚀 Como Usar

1. Em uma instância EC2, execute os seguintes comandos para instalar git e docker:

   ```bash
   sudo yum update -y
   sudo yum install git -y
   sudo yum install docker -y
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo curl -L "https://github.com/docker/compose/releases/download/$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d" -f4)/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. Crie a pasta .aws para inserir suas credenciais:
   ```bash
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install
   mkdir -p ~/.aws
   cd ~/.aws/
   nano config
   nano credentials
   ```
3. Clone o repositório:

   ```bash
   git clone "https://github.com/Compass-pb-aws-2024-ABRIL/sprints-4-5-pb-aws-abril.git"
   git checkout grupo-1
   ```

4. Crie um ambiente virtual:

   ```bash
   Em linux:
       pip install virtualenv virtualenvwrapper
       python3 .11 -m venv nome_do_ambiente
       source nome_do_ambiente/bin/activate

   Em Windows:
       pip install virtualenv virtualenvwrapper-win
       mkvirtualenv nome_do_ambiente -p python3.11
   ```

### Usando Python

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Execute a API:
   ```bash
   python src/api/app.py
   ```
5. Acesse a API:
   ```bash
   http://localhost:8000/docs
   ```

### Usando Docker

3. Construa a Imagem Docker:

   ```bash
   docker build -t nome-da-imagem .
   ```

4. Execute o Container Docker:
   ```bash
   docker run -d -p 8000:8000 nome-da-imagem
   ```
5. Acesse a API:
   ```bash
   http://localhost:8000/docs
   ```

---

### Uso da API

- A API estará disponível no IP público da instância EC2 na porta 8000:

  ```
  http://seu-ip-publico:8000
  ```

- Para fazer uma predição, envie uma requisição POST para o endpoint /api/v1/predict com os dados necessários, como o exemplo a seguir:

  ```
  {
  "no_of_adults": 2,
  "no_of_children": 1,
  "no_of_weekend_nights": 1,
  "no_of_week_nights": 0,
  "type_of_meal_plan": 0,
  "required_car_parking_space": 0,
  "room_type_reserved": 0,
  "lead_time": 3,
  "arrival_year": 2017,
  "arrival_month": 8,
  "arrival_date": 23,
  "market_segment_type": 4,
  "repeated_guest": 0,
  "no_of_previous_cancellations": 0,
  "no_of_previous_bookings_not_canceled": 0,
  "no_of_special_requests": 3,
  "booking_status": 1
  }
  ```

  A resposta é entregue no seguinte formato:

  ```
  {
  "predict": 2
  }
  ```

## Diagrama de Arquitetura AWS

Diagrama de arquitetura da aplicação na AWS.

![AWS API Architecture](assets/sprint4-5.jpg)

---

## ✅ Tecnologias utilizadas

- `AWS - Sagemaker`
- `AWS - EC2`
- `AWS - S3`
- `AWS - RDS`
- `FastAPI`
- `Anaconda`
- `Jupyter`
- `Python`
- `MySQL`

---

## ❌ Dificuldades

- Lidar com a integração entre SageMaker, S3 e RDS.
- Rodar SageMaker em local mode

---

## 👨‍💻 Autor

- [Luiz Fillipe Oliveira Morais](https://github.com/LuizFillipe1)

---
