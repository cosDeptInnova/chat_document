@{
  CondaEnv      = "cosmos"
  GlobalEnvFile = "config\cosmos.prod.env"

  Services = @(
    @{
      Name="modeloNegocio"; Path="modeloNegocio";
      Args=@("main:app","--reload","--host","0.0.0.0","--port","8000");
      EnvFile="";
      Port=8000
    },
    @{
      Name="chat_document"; Path="chat_document";
      Args=@("main:app","--reload","--host","0.0.0.0","--port","8100","--workers","2");
      EnvFile="";
      Port=8100
    },
    @{
      Name="auth_sso"; Path="auth_sso";
      Args=@("main:app","--reload","--host","0.0.0.0","--port","7100");
      EnvFile="config\auth_sso.env";
      Port=7100
    },
    @{
      Name="comp_docs"; Path="comp_docs";
      Args=@("app.app:app","--reload","--host","0.0.0.0","--port","8007");
      EnvFile="";
      Port=8007
    },
    @{
      Name="cosmos_mcp"; Path="cosmos_mcp";
      Args=@("main:app","--reload","--host","0.0.0.0","--port","8090");
      EnvFile="";
      Port=8090
    },
    @{
      Name="login"; Path="login";
      Args=@("main:app","--reload","--host","0.0.0.0","--port","7000");
      EnvFile=""; # si lo necesitas: "config\login.env"
      Port=7000
    },
    @{
      Name="ocr"; Path="ocr";
      Args=@("main:app","--reload","--host","0.0.0.0","--port","8010");
      EnvFile="config\ocr.env";
      Port=8010
    },
    @{
      Name="web_search"; Path="web_search";
      Args=@("main:app","--reload","--host","0.0.0.0","--port","8200");
      EnvFile="";
      Port=8200
    },
    @{
      Name="nlp"; Path="nlp";
      Args=@("main:app","--workers","2","--host","0.0.0.0","--port","5000");
      EnvFile="config\nlp.env";
      Port=5000
    },
    @{
      Name="legal_search"; Path="legal_search";
      Args=@("main:app","--reload","--host","0.0.0.0","--port","8201");
      EnvFile="";
      Port=8201
    }
  )

  StartOrder = @(
    "auth_sso","login","cosmos_mcp","nlp","ocr",
    "legal_search","web_search","chat_document","comp_docs","modeloNegocio"
  )
}