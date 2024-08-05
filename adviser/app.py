from fastapi import FastAPI

def make_app():

    app = FastAPI()

    @app.get("/health_check")
    async def get_health():
        return "ok"

    return app


app = make_app()
