import subprocess
import time
import sys
import os
import signal
import platform

class AppRunner:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None

    def check_dependencies(self):
        """Check if required dependencies are installed."""
        print("🔍 Checking dependencies...")

        #check backend dependencies
        try:
            import fastapi
            import uvicorn
            print("✅ Backend dependencies found")
        except ImportError:
            print("❌ Backend dependencies missing. Run: pip install -r backend/requirements.txt")
            return False
        
        #check frontend dependencies
        try:
            import streamlit
            import requests
            print("✅ Frontend dependencies found")
        except ImportError:
            print("❌ Frontend dependencies missing. Run: pip install -r frontend/requirements.txt")
            return False
        return True
        
    def check_models(self):
        """Check if model files exist"""
        print("🔍 Checking model files...")
        model_files = ["models/neural_network.pkl", "models/preprocessor.pkl"]
        
        for file_path in model_files:
            if os.path.exists(file_path):
                print(f"✅ Found: {file_path}")
            else:
                print(f"❌ Missing: {file_path}")
                print("💡 Please save your trained model and preprocessor first!")
                return False
        return True
    
    def run_backend(self):
        """Start the FastAPI backend"""
        print("📡 Starting FastAPI backend...")

        backend_dir=os.path.join(os.getcwd(), "backend")

        if platform.system() == "Windows":
            self.backend_process = subprocess.Popen([
                sys.executable, '-m', 'uvicorn', 'main:app', 
                '--reload', '--host', '0.0.0.0', '--port', '8000'
            ], cwd=backend_dir, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            self.backend_process = subprocess.Popen([
                sys.executable, '-m', 'uvicorn', 'main:app', 
                '--reload', '--host', '0.0.0.0', '--port', '8000'
            ], cwd=backend_dir, preexec_fn=os.setsid)
        return self.backend_process
    
    def run_frontend(self):
        """Start the Streamlit frontend"""
        print("🎨 Starting Streamlit frontend...")
        
        frontend_dir = os.path.join(os.getcwd(), 'frontend')
        
        if platform.system() == "Windows":
            self.frontend_process = subprocess.Popen([
                sys.executable, '-m', 'streamlit', 'run', 'app.py', 
                '--server.port', '8501', '--server.address', '0.0.0.0'
            ], cwd=frontend_dir, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            self.frontend_process = subprocess.Popen([
                sys.executable, '-m', 'streamlit', 'run', 'app.py', 
                '--server.port', '8501', '--server.address', '0.0.0.0'
            ], cwd=frontend_dir, preexec_fn=os.setsid)
            
        return self.frontend_process
    
    def wait_for_backend(seld, max_retries=30):
        """Wait for backend to be ready"""
        import requests

        print("⏳ Waiting for backend to start...")

        for i in range(max_retries):
            try:
                response=requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Backend is ready!")
                    return True
            except:
                pass
            time.sleep(1)
            print(f"   Attempt {i+1}/{max_retries}...")
        
        print("❌ Backend failed to start within timeout")
        return False
    
    def cleanup(self):
        """Clean up processes"""
        print("\n🛑 Shutting down...")

        if self.frontend_process:
            try:
                if platform.system() == "Windows":
                    self.frontend_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self.frontend_process.pid), signal.SIGTERM)
                print("✅ Frontend stopped")
            except:
                pass

        if self.backend_process:
            try:
                if platform.system() == "Windows":
                    self.backend_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self.backend_process.pid), signal.SIGTERM)
                print("✅ Backend stopped")
            except:
                pass
    
    def run(self):
        """Main run method"""
        print("🚀 Starting Fake Job Detector Application...")
        print("=" * 50)

        if not self.check_dependencies():
            return
        if not self.check_models():
            return
        
        try:
            self.run_backend()
            if not self.wait_for_backend():
                self.cleanup()
                return
            
            self.run_frontend()
            time.sleep(3)

            print("\n" + "=" * 50)
            print("✅ Application started successfully!")
            print("📡 Backend API: http://localhost:8000")
            print("🎨 Frontend App: http://localhost:8501")
            print("📚 API Docs: http://localhost:8000/docs")
            print("📊 API Health: http://localhost:8000/health")
            print("=" * 50)
            print("Press Ctrl+C to stop the application")

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

        except Exception as e:
            print(f"❌ Error starting application: {str(e)}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    runner=AppRunner()
    runner.run()
#     return {"status": model_status, "message": "Models are ready" if model_status == "loaded" else "Models not loaded"}