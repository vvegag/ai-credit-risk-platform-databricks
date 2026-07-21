"""Script de deploy do RAG Agent no Databricks Model Serving."""
import mlflow
import sys
sys.path.append('/Workspace/Users/valdomirovega@hotmail.com/ai-credit-risk-platform-databricks/10_rag_agent')

from src.rag_agent import RAGAgent


class RAGModel(mlflow.pyfunc.PythonModel):
    """Wrapper MLflow para o RAG Agent."""
    
    def load_context(self, context):
        """Carrega o agente RAG."""
        self.agent = RAGAgent()
    
    def predict(self, context, model_input):
        """
        Processa queries.
        
        Args:
            model_input: DataFrame com coluna 'query'
            
        Returns:
            Lista de respostas
        """
        queries = model_input['query'].tolist()
        respostas = []
        
        for query in queries:
            resultado = self.agent.query(query, verbose=False)
            respostas.append(resultado['resposta'])
        
        return respostas


def deploy_model():
    """Deploy do modelo no Unity Catalog + Model Serving."""
    
    print("🚀 Iniciando deploy do RAG Agent...\n")
    
    # Configurar MLflow
    mlflow.set_registry_uri("databricks-uc")
    
    # Nome do modelo no UC
    model_name = "credit_risk.models.rag_agent"
    
    # Logar modelo
    print("1. Logando modelo no MLflow...")
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="rag_agent",
            python_model=RAGModel(),
            registered_model_name=model_name,
            pip_requirements=[
                "sentence-transformers>=2.2.0",
                "langchain>=0.1.0",
                "langchain-databricks>=0.1.0",
                "databricks-vectorsearch>=0.22.0",
                "mlflow>=2.10.0"
            ]
        )
    
    print(f"✅ Modelo registrado: {model_name}\n")
    
    print("2. Próximos passos:")
    print("   a. Acesse: Machine Learning > Serving")
    print("   b. Clique em 'Create Endpoint'")
    print(f"   c. Selecione o modelo: {model_name}")
    print("   d. Configure compute: Serverless (recomendado)")
    print("   e. Deploy!\n")
    
    print("🎉 Deploy completo!")
    print(f"\nModel URI: {mlflow.get_artifact_uri()}")
    print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    deploy_model()