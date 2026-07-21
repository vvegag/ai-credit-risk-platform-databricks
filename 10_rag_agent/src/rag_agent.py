"""Módulo principal do RAG Agent."""
from typing import Dict, Any, List
from langchain_databricks import ChatDatabricks
from langchain.prompts import PromptTemplate
from .config import CONFIG
from .vector_search import VectorSearch


class RAGAgent:
    """Agente RAG completo para análise de documentos de crédito."""
    
    # Template do prompt
    PROMPT_TEMPLATE = """Você é um analista de crédito especializado em análise de documentos empresariais.

Sua tarefa é responder a pergunta do usuário baseando-se EXCLUSIVAMENTE nos documentos fornecidos abaixo.

**REGRAS IMPORTANTES:**
1. Use APENAS as informações presentes nos documentos
2. Se a informação não estiver nos documentos, diga "Não encontrei essa informação nos documentos disponíveis"
3. Cite sempre os documentos que você usou (ex: "Segundo o contrato_5...")
4. Seja preciso e objetivo
5. Use dados numéricos quando disponíveis
6. Responda em português brasileiro

**DOCUMENTOS RELEVANTES:**
{context}

**PERGUNTA DO USUÁRIO:**
{question}

**SUA RESPOSTA:**
"""
    
    def __init__(self, 
                 vector_search: VectorSearch = None,
                 llm_endpoint: str = None,
                 temperature: float = None,
                 max_tokens: int = None):
        """
        Inicializa o RAG Agent.
        
        Args:
            vector_search: Cliente de busca vetorial
            llm_endpoint: Endpoint do LLM
            temperature: Temperatura do LLM
            max_tokens: Máximo de tokens
        """
        # Vector Search
        self.vector_search = vector_search or VectorSearch()
        
        # LLM
        self.llm = ChatDatabricks(
            endpoint=llm_endpoint or CONFIG.llm.endpoint,
            temperature=temperature or CONFIG.llm.temperature,
            max_tokens=max_tokens or CONFIG.llm.max_tokens
        )
        
        # Prompt
        self.prompt = PromptTemplate(
            template=self.PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Chain RAG
        self.chain = self.prompt | self.llm
    
    def query(self, 
              question: str, 
              top_k: int = None,
              verbose: bool = False) -> Dict[str, Any]:
        """
        Processa uma pergunta usando RAG.
        
        Args:
            question: Pergunta do usuário
            top_k: Número de documentos a recuperar
            verbose: Logs detalhados
            
        Returns:
            Dict com resposta e metadados
        """
        top_k = top_k or CONFIG.top_k_documents
        
        if verbose:
            print(f"🤖 RAG Agent processando: {question}\n")
        
        # 1. Busca vetorial
        documentos = self.vector_search.search(question, top_k=top_k)
        
        if not documentos:
            return {
                'resposta': "Não encontrei documentos relevantes.",
                'documentos': [],
                'erro': 'Nenhum documento encontrado'
            }
        
        if verbose:
            print(f"✅ {len(documentos)} documentos recuperados\n")
        
        # 2. Construir contexto
        context = self._build_context(documentos)
        
        # 3. Gerar resposta
        try:
            resposta = self.chain.invoke({
                "context": context,
                "question": question
            }).content
            
            return {
                'resposta': resposta,
                'documentos': documentos,
                'num_documentos': len(documentos),
                'pergunta': question
            }
            
        except Exception as e:
            return {
                'resposta': f"Erro ao gerar resposta: {e}",
                'documentos': documentos,
                'erro': str(e)
            }
    
    def _build_context(self, documentos: List[Dict[str, Any]]) -> str:
        """
        Constrói contexto a partir dos documentos.
        
        Args:
            documentos: Lista de documentos recuperados
            
        Returns:
            Contexto formatado
        """
        context_parts = []
        for i, doc in enumerate(documentos, 1):
            context_parts.append(
                f"[DOCUMENTO {i}]\n"
                f"ID: {doc.get('documento_id', 'N/A')}\n"
                f"Tipo: {doc.get('tipo_documento', 'N/A')}\n"
                f"Cliente: {doc.get('id_cliente', 'N/A')}\n"
                f"Conteúdo:\n{doc.get('texto', '')}\n"
            )
        
        return "\n" + "-"*80 + "\n".join(context_parts)