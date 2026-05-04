create table if not exists agent_router_logs (
  id bigserial primary key,
  session_id text not null,
  turn_index integer,
  user_message text not null,
  language text,
  intent text not null,
  keywords jsonb not null default '[]'::jsonb,
  english_translation_or_summary text,
  router_model text not null,
  router_base_url text,
  router_payload jsonb not null,
  created_at timestamptz not null default now(),
  latency_ms integer not null check (latency_ms >= 0)
);

alter table agent_router_logs enable row level security;

create index if not exists idx_agent_router_logs_session
on agent_router_logs (session_id, turn_index);

create index if not exists idx_agent_router_logs_intent
on agent_router_logs (intent);
