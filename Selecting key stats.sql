SELECT p.player_id, full_name, game_date, matchup, l.min, pts, reb, ast FROM public.player_game_logs as l
join public.players p on p.player_id = l.player_id
order by game_date desc