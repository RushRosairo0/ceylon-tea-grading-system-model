echo "================================"
echo "python model API - development"
echo "port: 8001"
echo "================================"
echo ""

# activate virtual environment
source venv/bin/activate
echo "virtual environment activated"
echo "starting server..."
echo ""

# show API url
echo "API running at http://localhost:8001/docs"
echo ""

# Run FastAPI server
uvicorn api.inference_api:app --host 0.0.0.0 --port 8001 --reload

echo ""
echo "================================"
echo "python model API stopped"
echo "================================"
