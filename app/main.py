from fastapi import FastAPI

from app.routers import code, data, finance, geo, image, signals, tasks, text, workflows


def create_app() -> FastAPI:
    app = FastAPI(
        title="ToduFodu Backend",
        description=(
            "High-octane API suite delivering heavy-duty data, text, image, "
            "and analytics workloads for ambitious frontends."
        ),
        version="0.1.0",
    )

    app.include_router(text.router, prefix="/text", tags=["text"])
    app.include_router(data.router, prefix="/data", tags=["data"])
    app.include_router(finance.router, prefix="/finance", tags=["finance"])
    app.include_router(image.router, prefix="/image", tags=["image"])
    app.include_router(geo.router, prefix="/geo", tags=["geo"])
    app.include_router(signals.router, prefix="/signals", tags=["signals"])
    app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
    app.include_router(code.router, prefix="/code", tags=["code"])
    app.include_router(workflows.router, prefix="/workflows", tags=["workflows"])

    return app


app = create_app()

