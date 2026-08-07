begin;

alter table public.weather_daily add column if not exists data_kind text not null default 'measured';
alter table public.weather_daily add column if not exists checked boolean not null default false;
alter table public.weather_daily add column if not exists check_message text;
alter table public.weather_daily add column if not exists humidity_avg_pct numeric;
alter table public.weather_daily add column if not exists precipitation_mm numeric;
alter table public.weather_daily add column if not exists et0_mm numeric;

create table if not exists public.app_settings (
    key text primary key,
    value text not null,
    updated_at timestamptz not null default now()
);

alter table public.app_settings enable row level security;
revoke all on public.app_settings from anon, authenticated;

commit;
