from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Usar 192.168.13.22 ya que el contenedor de PostgreSQL está expuesto en el puerto 5432
DATABASE_URL = "postgresql://admin:adminC0smo5.GDC@localhost:5432/cosmos_control"

# Crear motor de conexión a PostgreSQL
engine = create_engine(DATABASE_URL)

# Crear una clase base para los modelos ORM
Base = declarative_base()

# Crear una clase de sesión
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependencia para crear y cerrar sesiones
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
