import enum
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Enum as SAEnum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import settings
from app.database import Base


class ImageType(str, enum.Enum):
    """Tipos de imagem de um produto de joalheria."""

    clean = "clean"          # fundo branco  (_fundobranco)
    environment = "environment"  # ambiente      (_ambiente)
    person = "person"        # com pessoa    (_pessoa)


class Product(Base):
    """
    Produto de joalheria identificado pelo product_id extraído do nome do arquivo.

    O embedding é gerado exclusivamente da imagem 'clean' (fundo branco)
    para garantir consistência na busca por similaridade.
    """

    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    main_image_url: Mapped[str] = mapped_column(String(2048), nullable=False)  # chave S3 da imagem clean
    embedding: Mapped[list] = mapped_column(Vector(settings.EMBEDDING_DIM), nullable=False)

    images: Mapped[list["ProductImage"]] = relationship(
        "ProductImage", back_populates="product", cascade="all, delete-orphan"
    )


class ProductImage(Base):
    """
    Imagem individual de um produto, associada por product_id.

    Cada produto pode ter múltiplas imagens de tipos distintos
    (clean, environment, person).
    """

    __tablename__ = "product_images"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    product_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True
    )
    type: Mapped[ImageType] = mapped_column(SAEnum(ImageType, name="imagetype"), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)  # chave S3

    product: Mapped["Product"] = relationship("Product", back_populates="images")
