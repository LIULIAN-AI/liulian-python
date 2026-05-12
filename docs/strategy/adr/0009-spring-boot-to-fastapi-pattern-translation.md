# ADR 0009 — Spring Boot patterns translate to FastAPI for `liulian-api`

- **Status**: accepted
- **Decision date**: 2026-05-12

## Context

`neobanker-backend-MVP-V2` is Java / Spring Boot 3.1 with 44+
`@RestController`s, 70+ `@Entity` classes, 49 repositories, 31 DB tables,
JPA + Elasticsearch + Redis + MinIO + Caffeine.

LIULIAN's backend (`liulian-api`) is Python / FastAPI. We cannot fork
the Spring Boot code, but we **borrow the patterns** verbatim.

## Decision

Translate (not fork) the neobanker backend patterns to FastAPI +
Pydantic v2 + SQLModel. Specifically:

| Spring Boot pattern | FastAPI translation |
|---|---|
| `@RestController` + `@RequestMapping` | `APIRouter(prefix=…, tags=…)` |
| `@GetMapping("/{id}")` `getById` | `@router.get("/{id}", response_model=…)` |
| `@PostMapping` + `@RequestBody @Valid` | Pydantic body model |
| `Page<T>` + `Pageable` | `Page[T]` generic with `{items, total, page, page_size}` |
| `@Service` + `@Repository` | `services/*.py` + `repositories/*.py` (kept symmetrical for legibility) |
| `application.yml` profiles | `liulian_api/config.py` per-environment via env vars |
| `@Valid` exception envelope | RFC-7807-ish `{code, message, details}` |
| `spring-boot-starter-actuator` `/actuator/health` | FastAPI `/healthz` + `/readyz` |
| `spring-boot-starter-data-jpa` `@Entity` | SQLModel (one class for DB + API) |
| `spring-boot-starter-data-redis` | `redis-py` + `arq` workers |
| `spring-boot-starter-mail` | `aiosmtplib` |
| `mapstruct` DTO ↔ Entity mappers | NOT used; Pydantic `model_validate` / `model_dump` |
| `lombok` getters/setters | NOT used; Python dataclasses + Pydantic |
| `restart_neobanker.sh` | `liulianctl restart api` |

## What we DROP

- **Mapstruct** + **Lombok** ceremony: Java-specific verbosity that
  collapses into Pydantic v2 in Python.
- **Spring profile bloat**: env vars only; no profile activation.
- **Bean discovery / dependency injection magic**: explicit FastAPI
  `Depends(...)` calls.

## What we KEEP (preserved contracts)

- **Pagination shape**: `{items, total, page, page_size}` verbatim.
- **Error envelope shape**: `{code, message, details}` verbatim.
- **Audit-log row schema**: same fields (`actor_id`, `action`,
  `target_kind`, `target_id`, `metadata`, `at`).
- **CORS env-var name**: `LIULIAN_API_CORS_ALLOWED_ORIGINS` (mirrors
  `AGENT_CORS_ALLOWED_ORIGINS` in neobanker).
- **JWT verification middleware shape**: Spring Filter → FastAPI
  middleware, same claims structure.

## Rationale

- **Cross-product developer onboarding**: a developer who worked on
  neobanker's backend can read `liulian-api` and know where everything
  lives.
- **Avoid Java verbosity**: Python's compact types + Pydantic v2 do in
  one class what Spring requires three for (Entity / DTO / Mapper).
- **Same observability**: actuator-style health endpoints translate
  directly; metrics + traces look the same in Grafana.

## Alternatives considered

- **Direct Java port** (Jakarta EE on JBoss / WildFly): rejected;
  Python is the research-core language and ML ecosystem.
- **GraphQL** (Hasura / PostGraphile): rejected for M1; can be added at
  M3 as an alternative read API.
- **Falcon / Starlette without FastAPI**: rejected; FastAPI's OpenAPI
  auto-gen is load-bearing for the SDK + frontend codegen story.

## Consequences

- (+) Familiar conventions for anyone with Spring Boot background.
- (+) OpenAPI schema flows directly to SDK + web + mobile codegen.
- (−) FastAPI's plugin ecosystem is smaller than Spring's; we may
  hand-roll some middleware (e.g. structured request log).

## Cross-references

- `NEOBANKER_REUSE_MAP.md` §14.1 (the full translation table)
- `PLATFORM_BLUEPRINT.md` §5 (backend stack)
