# 🚀 GUIA RÁPIDO: Publicar ai-credit-risk-platform no GitHub

## ✅ PASSO A PASSO (10 minutos)

### 1️⃣ Renomear Pasta no Databricks (2 min)

**No Databricks Workspace:**
1. Navegue até `/Users/valdomirovega@hotmail.com/`
2. **Clique com botão direito** em `projeto_inadimplencia`
3. Selecione **"Rename"**
4. Digite: `ai-credit-risk-platform`
5. ✅ Confirme

---

### 2️⃣ Criar Repositório no GitHub (2 min)

1. Acesse: https://github.com/vvegag
2. Clique em **"New repository"** (botão verde)
3. Preencha:
   - **Repository name**: `ai-credit-risk-platform`
   - **Description**: `End-to-end ML platform for credit risk prediction using Databricks`
   - **Public** ✅ (recomendado para portfolio)
   - **NÃO** marque "Initialize this repository with a README"
4. Clique em **"Create repository"**
5. **COPIE a URL**: `https://github.com/vvegag/ai-credit-risk-platform.git`

---

### 3️⃣ Gerar Personal Access Token (PAT) no GitHub (2 min)

1. Acesse: https://github.com/settings/tokens
2. Clique em **"Generate new token"** → **"Generate new token (classic)"**
3. Preencha:
   - **Note**: `Databricks Access`
   - **Expiration**: `No expiration` (ou 90 days se preferir)
   - **Select scopes**:
     - ☑️ **repo** (marque TODOS os sub-itens)
     - ☑️ **workflow**
4. Role até o final e clique em **"Generate token"**
5. **⚠️ IMPORTANTE**: Copie o token AGORA (formato: `ghp_xxxxxxxxxxxx`)
   - Você só verá uma vez!
   - Cole em um arquivo temporário

---

### 4️⃣ Configurar Git no Databricks (2 min)

**Opção A: Via Interface (Mais Fácil)**
1. No Databricks, clique no seu **avatar** (canto superior direito)
2. **Settings** → **User Settings** → **Git Integration**
3. Clique em **"Add Git Credential"**
4. Selecione **"GitHub"**
5. Preencha:
   - **Git username**: `vvegag`
   - **Token**: Cole o PAT que você copiou no passo 3
6. Clique em **"Save"**

---

### 5️⃣ Substituir README Antigo (1 min)

```bash
# Execute este código em um notebook Databricks:
%sh
cd /Workspace/Users/valdomirovega@hotmail.com/ai-credit-risk-platform

# Backup do README antigo
mv README.md README_OLD.md

# Usar o novo README
mv README_NEW.md README.md

echo "✅ README atualizado!"
```

---

### 6️⃣ Inicializar Git e Fazer Push (3 min)

```bash
# Execute no notebook:
%sh
cd /Workspace/Users/valdomirovega@hotmail.com/ai-credit-risk-platform

# Executar o script de setup
chmod +x setup_github.sh
./setup_github.sh

# Quando terminar, execute o push:
git push -u origin main
```

**ATENÇÃO no Push:**
- Quando solicitar **Username**: `vvegag`
- Quando solicitar **Password**: Cole seu **Personal Access Token** (não a senha do GitHub!)

---

## ✅ CHECKLIST FINAL

Após completar os passos acima:

- [ ] Pasta renomeada para `ai-credit-risk-platform`
- [ ] Repositório criado no GitHub
- [ ] Personal Access Token gerado e salvo
- [ ] Credencial Git configurada no Databricks
- [ ] README novo substituindo o antigo
- [ ] Git inicializado com `setup_github.sh`
- [ ] Push realizado com sucesso
- [ ] Repositório visível em: `https://github.com/vvegag/ai-credit-risk-platform`

---

## 🎯 Verificar se Funcionou

1. Acesse: https://github.com/vvegag/ai-credit-risk-platform
2. Você deve ver:
   - ✅ README com badges coloridos
   - ✅ Todas as pastas (01_setup, 02_ingestion, etc.)
   - ✅ Arquivos (.gitignore, LICENSE, etc.)

---

## 🐛 Troubleshooting

### Erro: "remote: Support for password authentication was removed"
**Solução**: Você está usando a senha do GitHub em vez do PAT. Use o **Personal Access Token** como senha.

### Erro: "fatal: not a git repository"
**Solução**: Execute o script `setup_github.sh` primeiro.

### Erro: "refusing to merge unrelated histories"
**Solução**: O repositório não deve estar inicializado com README. Delete e recrie sem README inicial.

---

## 🎉 PRONTO!

Seu repositório está no ar! Agora você pode:

1. **Adicionar ao LinkedIn**:
   - Portfolio → Adicionar projeto
   - Link: https://github.com/vvegag/ai-credit-risk-platform

2. **Compartilhar**:
   - Tweet sobre o projeto
   - Post no LinkedIn
   - Adicionar ao currículo

3. **Melhorar**:
   - Adicionar screenshots do dashboard em `09_docs/images/`
   - Criar diagramas de arquitetura
   - Adicionar badges de CI/CD futuramente

---

**🚀 Boa sorte e sucesso nas entrevistas!**
