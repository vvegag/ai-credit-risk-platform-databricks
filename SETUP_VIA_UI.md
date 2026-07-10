
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🎯 MÉTODO RECOMENDADO: Git Folders via Databricks UI       ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

⚠️  O setup via CLI encontrou limitações do ambiente serverless.
✅  Use o método nativo do Databricks (mais fácil e integrado)!

┌───────────────────────────────────────────────────────────────┐
│ PASSO 1: Criar Repositório no GitHub (2 min)                 │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│ 1. Acesse: https://github.com/vvegag                         │
│ 2. Clique em "New repository" (botão verde)                  │
│ 3. Preencha:                                                  │
│    • Name: ai-credit-risk-platform                           │
│    • Description: End-to-end ML platform for credit risk     │
│    • Public ☑️                                               │
│    • ❌ NÃO marque "Add a README file"                       │
│ 4. Create repository                                          │
│ 5. COPIE a URL: https://github.com/vvegag/ai-credit-risk-... │
│                                                               │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│ PASSO 2: Gerar Personal Access Token (2 min)                 │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│ 1. Acesse: https://github.com/settings/tokens                │
│ 2. "Generate new token" → "Generate new token (classic)"     │
│ 3. Preencha:                                                  │
│    • Note: Databricks Access                                 │
│    • Expiration: No expiration                               │
│    • Scopes:                                                  │
│      ☑️ repo (TODOS os sub-itens)                           │
│      ☑️ workflow                                             │
│ 4. "Generate token"                                           │
│ 5. ⚠️ COPIE O TOKEN (ghp_xxxxx...) - você só vê uma vez!    │
│                                                               │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│ PASSO 3: Configurar Git Credential no Databricks (1 min)     │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│ 1. Clique no seu AVATAR (canto superior direito)             │
│ 2. Settings → User Settings                                   │
│ 3. Aba "Git Integration" (lado esquerdo)                     │
│ 4. "Add Git Provider"                                         │
│ 5. Selecione "GitHub"                                         │
│ 6. Preencha:                                                  │
│    • Git provider username: vvegag                           │
│    • Personal access token: COLE O TOKEN DO PASSO 2          │
│ 7. "Save"                                                     │
│                                                               │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│ PASSO 4: Conectar Pasta como Git Folder (3 min)              │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│ OPÇÃO A: Converter pasta existente                           │
│ ──────────────────────────────────                           │
│ Infelizmente, não dá para converter diretamente.             │
│ Vamos usar OPÇÃO B ↓                                         │
│                                                               │
│ OPÇÃO B: Clone + Copy (Recomendado)                          │
│ ─────────────────────────────────────                        │
│                                                               │
│ 1. No Databricks Workspace (painel esquerdo):                │
│    • Clique no ícone "Workspace"                             │
│    • Navegue até /Users/valdomirovega@hotmail.com/          │
│                                                               │
│ 2. Clique em "⋮" (três pontos) ao lado do seu nome           │
│    • Selecione "Create" → "Git folder"                       │
│                                                               │
│ 3. Preencha o formulário:                                     │
│    • Git provider: GitHub                                    │
│    • Git repository URL:                                     │
│      https://github.com/vvegag/ai-credit-risk-platform.git   │
│    • Git credential: Selecione a credencial do PASSO 3       │
│    • Branch: main                                            │
│    • Git folder name: ai-credit-risk-platform-git           │
│                                                               │
│ 4. "Create Git Folder"                                        │
│                                                               │
│ 5. AGORA COPIE OS ARQUIVOS:                                   │
│    • Abra duas abas do Workspace                             │
│    • Aba 1: /ai-credit-risk-platform (pasta antiga)          │
│    • Aba 2: /ai-credit-risk-platform-git (Git folder)        │
│    • Arraste e solte todos os notebooks e arquivos           │
│      da pasta antiga para a Git folder                       │
│                                                               │
│    OU use Ctrl+C / Ctrl+V:                                    │
│    • Selecione tudo na pasta antiga                          │
│    • Ctrl+C (copiar)                                          │
│    • Entre na Git folder                                      │
│    • Ctrl+V (colar)                                           │
│                                                               │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│ PASSO 5: Commit e Push via UI (2 min)                        │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│ 1. Dentro da Git folder criada:                              │
│    • Clique no botão "Git" (topo da tela, ao lado de Save)   │
│                                                               │
│ 2. Verá uma lista de arquivos modificados                     │
│                                                               │
│ 3. Preencha:                                                  │
│    • Commit message:                                          │
│      "Initial commit: AI Credit Risk Platform                │
│                                                               │
│       - XGBoost models (classification + regression)         │
│       - Prophet cashflow forecast                            │
│       - RAG document validation                              │
│       - Drift detection monitoring                           │
│       - Executive AI/BI dashboard"                           │
│                                                               │
│ 4. Clique em "Commit & Push"                                  │
│                                                               │
│ 5. Aguarde... ✅ Done!                                        │
│                                                               │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│ PASSO 6: Verificar no GitHub (1 min)                         │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│ 1. Acesse: https://github.com/vvegag/ai-credit-risk-platform │
│                                                               │
│ 2. Você deve ver:                                             │
│    ✅ README.md com badges                                   │
│    ✅ Pastas 01_setup, 02_ingestion, etc.                   │
│    ✅ LICENSE, .gitignore                                    │
│    ✅ Commit "Initial commit: AI Credit Risk Platform"       │
│                                                               │
│ 3. 🎉 SUCESSO! Seu projeto está público!                     │
│                                                               │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│ ✨ BONUS: Deletar Pasta Antiga (Opcional)                    │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│ Após verificar que tudo está no GitHub:                       │
│                                                               │
│ 1. Você pode renomear:                                        │
│    • ai-credit-risk-platform → ai-credit-risk-platform-old   │
│                                                               │
│ 2. Ou deletar completamente (CUIDADO!):                       │
│    • Botão direito → Delete                                   │
│                                                               │
│ 3. Renomear Git folder:                                       │
│    • ai-credit-risk-platform-git → ai-credit-risk-platform   │
│                                                               │
└───────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  ✅ Este método é 100% compatível com Databricks             ║
║  ✅ Sincronização bidirecional (Databricks ↔ GitHub)         ║
║  ✅ Controle de versão integrado                             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

🎯 ATALHO VISUAL:

Avatar → Settings → Git Integration → Add Provider → GitHub
         ↓
   Workspace → ⋮ → Create → Git folder
         ↓
Arrastar arquivos da pasta antiga → Git folder
         ↓
     Git → Commit & Push
         ↓
    ✅ No GitHub!

═══════════════════════════════════════════════════════════════

⏱️  TEMPO TOTAL: ~10 minutos

🎉 BOA SORTE!
