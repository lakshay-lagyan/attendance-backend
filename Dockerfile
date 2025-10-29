# Stage 1: Builder - compile dependencies
FROM python:3.11-slim-bookworm AS builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    cmake \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip==23.3.1 setuptools==69.0.2 wheel==0.42.0

# Copy constraints file
COPY constraints.txt /tmp/constraints.txt

# Set pip to always use constraints
ENV PIP_CONSTRAINT=/tmp/constraints.txt

# CRITICAL: Install NumPy 1.x FIRST and lock it
RUN pip install --no-cache-dir "numpy==1.24.3"

# Install Flask ecosystem
RUN pip install --no-cache-dir --timeout=300 \
    Flask==3.0.0 \
    Werkzeug==3.0.1 \
    gunicorn==21.2.0 \
    python-dotenv==1.0.0

# Install database packages
RUN pip install --no-cache-dir --timeout=300 \
    SQLAlchemy==2.0.23 \
    psycopg2-binary==2.9.9 \
    Flask-SQLAlchemy==3.1.1 \
    Flask-Migrate==4.0.5 \
    alembic==1.12.1

# Install Redis packages
RUN pip install --no-cache-dir --timeout=300 \
    redis==5.0.1 \
    hiredis==2.2.3 \
    Flask-Caching==2.1.0

# Install auth packages
RUN pip install --no-cache-dir --timeout=300 \
    Flask-JWT-Extended==4.5.3 \
    Flask-Bcrypt==1.0.1 \
    bcrypt==4.1.1 \
    Flask-CORS==4.0.0 \
    Flask-Limiter==3.5.0 \
    Flask-Compress==1.14

# Install image processing (with --no-deps to prevent numpy upgrade)
RUN pip install --no-cache-dir --timeout=600 Pillow==10.1.0

# Install OpenCV (force no numpy upgrade)
RUN pip install --no-cache-dir --timeout=600 --no-deps opencv-python-headless==4.8.1.78 && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78

# Install TensorFlow with numpy constraint
RUN pip install --no-cache-dir --timeout=900 \
    "tensorflow==2.15.0" \
    "tf-keras==2.15.0"

# Verify and force NumPy 1.24.3 again
RUN pip install --no-cache-dir --force-reinstall "numpy==1.24.3"

# Install FAISS (requires NumPy 1.x)
RUN pip install --no-cache-dir --timeout=300 --no-deps faiss-cpu==1.7.4 && \
    pip install --no-cache-dir faiss-cpu==1.7.4

# Install face recognition packages
RUN pip install --no-cache-dir --timeout=300 \
    deepface==0.0.79 \
    mtcnn==0.1.1

# FINAL: Force reinstall NumPy 1.24.3 to override any upgrades
RUN pip install --no-cache-dir --force-reinstall --no-deps "numpy==1.24.3"

# Verify NumPy version
RUN python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); assert numpy.__version__.startswith('1.24'), 'Wrong NumPy version!'"

# Install utility packages
RUN pip install --no-cache-dir --timeout=300 \
    requests==2.31.0 \
    python-dateutil==2.8.2 \
    pytz==2023.3

# Stage 2: Runtime
FROM python:3.11-slim-bookworm

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgfortran5 \
    # PostgreSQL client library
    libpq5 \
    # Health check utility
    curl \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with specific UID for security
RUN groupadd -r -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -m -s /bin/bash appuser && \
    mkdir -p /app /tmp/uploads /var/log/app && \
    chown -R appuser:appuser /app /tmp/uploads /var/log/app

# Copy virtual environment from builder stage
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

# Ensure all Python files are readable
RUN find /app -type f -name "*.py" -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \;

# Switch to non-root user
USER appuser

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app \
    PORT=10000 \
    # TensorFlow optimizations
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_USE_LEGACY_KERAS=0 \
    # NumPy compatibility
    NPY_DISABLE_CPU_FEATURES="" \
    OPENBLAS_NUM_THREADS=1

# Verify NumPy version at runtime
RUN python -c "import numpy; print(f'Runtime NumPy: {numpy.__version__}')"

# Expose application port
EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

CMD ["gunicorn", \
     "main:app", \
     "--bind", "0.0.0.0:10000", \
     "--workers", "2", \
     "--threads", "4", \
     "--worker-class", "gthread", \
     "--worker-tmp-dir", "/dev/shm", \
     "--timeout", "300", \
     "--graceful-timeout", "30", \
     "--keep-alive", "5", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--capture-output", \
     "--enable-stdio-inheritance", \
     "--preload"]