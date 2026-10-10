param(
    [Parameter(Mandatory = $true)][string]$Project,
    [string]$Region = "us-central1",
    [string]$Service = "wireless-log-triage",
    [string]$OllamaUrl = ""
)

if (-not (Test-Path -LiteralPath "data/models.joblib")) {
    throw "Run wireless-triage prepare before deployment."
}
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "Google Cloud CLI (gcloud) is required."
}

$Image = "$Region-docker.pkg.dev/$Project/wireless-triage/$Service`:latest"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project $Project
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
gcloud artifacts repositories describe wireless-triage --location $Region --project $Project *> $null
if ($LASTEXITCODE -ne 0) {
    gcloud artifacts repositories create wireless-triage --repository-format docker --location $Region --project $Project
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
gcloud builds submit --tag $Image --project $Project .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$Environment = "TRIAGE_DATA_DIR=/data"
if ($OllamaUrl) { $Environment += ",TRIAGE_OLLAMA_URL=$OllamaUrl" }
gcloud run deploy $Service --image $Image --region $Region --platform managed --allow-unauthenticated --memory 4Gi --cpu 2 --max-instances 1 --concurrency 1 --set-env-vars $Environment --project $Project
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
