.PHONY: install api web dev test docker stop

# Install backend (editable) + frontend deps
install:
	pip install -e ".[dev]"
	cd web && pnpm install

# Backend only — http://localhost:8000/docs
api:
	. .venv/bin/activate && uvicorn api.main:app --reload --port 8000

# Frontend only — http://localhost:3000
web:
	cd web && pnpm dev --port 3000

# Run backend + frontend together (Ctrl-C stops both)
dev:
	@echo "API  → http://localhost:8000  ·  WEB → http://localhost:3000"
	@trap 'kill 0' INT TERM; \
	( . .venv/bin/activate && uvicorn api.main:app --reload --port 8000 ) & \
	( cd web && pnpm dev --port 3000 ) & \
	wait

# Backend test suite (177 tests)
test:
	. .venv/bin/activate && python3 -m pytest tests/ -q

# One-command containerized stack
docker:
	docker compose up --build

stop:
	docker compose down
