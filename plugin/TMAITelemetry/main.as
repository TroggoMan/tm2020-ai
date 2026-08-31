// TMAI Telemetry - serves live vehicle + race state as JSON lines on
// 127.0.0.1:8766, so the external driving policy can read real telemetry
// without having to be a plugin itself.
//
// Requires Openplanet Signature Mode = School (for the VehicleState API).
// Companion client: telemetry/telemetry_listener.py
//
// The plugin is the SERVER and Python is the client, matching the
// TrackmaniaRL Connect plugin's model: the policy process can restart and
// reconnect freely without the plugin needing any reconnect logic.

[Setting name="Listen port" min=1024 max=65535]
uint S_Port = 8766;

[Setting name="Max send rate (Hz), 0 = every frame" min=0 max=1000]
uint S_RateHz = 100;

[Setting name="Include per-wheel detail"]
bool S_Wheels = true;

[Setting name="Include every vehicle in the scene (for ghost capture)"]
bool S_AllVehicles = true;

[Setting name="Emit one record per local player (splitscreen)"]
bool S_Players = true;

Net::Socket@ g_server = null;
bool g_clientConnected = false;
uint g_lines = 0;

void OnDestroyed() { Shutdown(); }
void OnDisabled()  { Shutdown(); }

void Shutdown() {
    if (g_server !is null) {
        g_server.Close();
        @g_server = null;
    }
    g_clientConnected = false;
}

// Renders a float without AngelScript's %g scientific notation, which would
// still be valid JSON but is painful to eyeball in the raw stream.
string F(float v) { return Text::Format("%.5f", v); }

string V3(const vec3 &in v) {
    return "[" + F(v.x) + "," + F(v.y) + "," + F(v.z) + "]";
}

// The race clock. CSmArenaRulesMode::Now is authoritative when the mode
// script is up; fall back to the network game time when it isn't.
int RaceTimeMs(CSmScriptPlayer@ api) {
    if (api is null || api.StartTime <= 0) return 0;
    auto app = cast<CTrackMania>(GetApp());
    if (app is null) return 0;
    auto rules = cast<CSmArenaRulesMode>(app.PlaygroundScript);
    int64 now = rules is null
        ? int64(app.Network.PlaygroundClientScriptAPI.GameTime)
        : int64(rules.Now);
    return now > api.StartTime ? int(now - api.StartTime) : 0;
}

// Wheel order is FL, FR, RL, RR everywhere in this file, including the arrays
// the policy consumes. VehicleState::GetWheelDirt uses the same indexing.
string WheelsJson(CSceneVehicleVisState@ v) {
    return ",\"slip\":[" + F(v.FLSlipCoef) + "," + F(v.FRSlipCoef) + ","
                         + F(v.RLSlipCoef) + "," + F(v.RRSlipCoef) + "]"
         + ",\"mat\":[" + int(v.FLGroundContactMaterial) + "," + int(v.FRGroundContactMaterial) + ","
                        + int(v.RLGroundContactMaterial) + "," + int(v.RRGroundContactMaterial) + "]"
         + ",\"damper\":[" + F(v.FLDamperLen) + "," + F(v.FRDamperLen) + ","
                           + F(v.RLDamperLen) + "," + F(v.RRDamperLen) + "]"
         // Grip-relevant per-wheel state the policy never had. Icing and wear
         // change how much of a steering input actually reaches the road, so
         // without them the same action has different effects for no visible
         // reason - which is exactly the sort of thing that stalls learning.
         + ",\"icing\":[" + F(v.FLIcing01) + "," + F(v.FRIcing01) + ","
                          + F(v.RLIcing01) + "," + F(v.RRIcing01) + "]"
         + ",\"wear\":[" + F(v.FLTireWear01) + "," + F(v.FRTireWear01) + ","
                         + F(v.RLTireWear01) + "," + F(v.RRTireWear01) + "]"
         + ",\"brake_coef\":[" + F(v.FLBreakNormedCoef) + "," + F(v.FRBreakNormedCoef) + ","
                               + F(v.RLBreakNormedCoef) + "," + F(v.RRBreakNormedCoef) + "]"
         + ",\"dirt\":[" + F(VehicleState::GetWheelDirt(v, 0)) + ","
                         + F(VehicleState::GetWheelDirt(v, 1)) + ","
                         + F(VehicleState::GetWheelDirt(v, 2)) + ","
                         + F(VehicleState::GetWheelDirt(v, 3)) + "]"
         + ",\"steer_angle\":" + F(v.FLSteerAngle);
}

// Active effects on the car right now.
//
// This is the "why is the car doing that" channel: turbo, reactor boost,
// cruise control, slow motion. Everything here is a documented member or a
// VehicleState export - no Dev:: raw-memory reads, which are not available to
// an unsigned plugin in School mode.
string EffectsJson(CSceneVehicleVisState@ v) {
    return ",\"turbo_time\":" + F(v.TurboTime)
         + ",\"turbo_lvl\":" + int(VehicleState::GetLastTurboLevel(v))
         + ",\"reactor_lvl\":" + int(v.ReactorBoostLvl)
         + ",\"reactor_type\":" + int(v.ReactorBoostType)
         + ",\"reactor_timer\":" + F(VehicleState::GetReactorFinalTimer(v))
         + ",\"cruise\":" + VehicleState::GetCruiseDisplaySpeed(v)
         // 1.0 is normal speed; the slow-motion gates step it down.
         + ",\"sim_coef\":" + F(v.SimulationTimeCoef)
         + ",\"vehicle\":" + int(VehicleState::GetVehicleType(v))
         + ",\"side_speed\":" + F(VehicleState::GetSideSpeed(v))
         + ",\"air_brake\":" + F(v.AirBrakeNormed)
         + ",\"wetness\":" + F(v.WetnessValue01)
         + ",\"water\":" + F(v.WaterImmersionCoef);
}

// Every vehicle in the scene, not just the viewed one. This is how a ghost
// racing alongside you gets captured - GetAllVis sees it even though
// ViewingPlayerState() only ever returns the camera's car.
string VehiclesJson() {
    auto scene = GetApp().GameScene;
    if (scene is null) return "";
    auto all = VehicleState::GetAllVis(scene);
    if (all.Length == 0) return "";

    string s = ",\"vehicles\":[";
    for (uint i = 0; i < all.Length; i++) {
        auto v = all[i].AsyncState;
        if (v is null) continue;
        if (i > 0) s += ",";
        s += "{\"pos\":" + V3(v.Position)
           + ",\"vel\":" + V3(v.WorldVel)
           + ",\"dir\":" + V3(v.Dir)
           + ",\"up\":" + V3(v.Up)
           + ",\"left\":" + V3(v.Left)
           + ",\"speed\":" + F(v.FrontSpeed)
           + ",\"gear\":" + v.CurGear
           + ",\"in_steer\":" + F(v.InputSteer)
           + ",\"in_gas\":" + F(v.InputGasPedal)
           + ",\"in_brake\":" + F(v.InputBrakePedal)
           + ",\"ground\":" + (v.IsGroundContact ? "true" : "false")
           + ",\"slip\":[" + F(v.FLSlipCoef) + "," + F(v.FRSlipCoef) + ","
                           + F(v.RLSlipCoef) + "," + F(v.RRSlipCoef) + "]"
           + "}";
    }
    return s + "]";
}

string BuildLine() {
    // Read the vehicle FIRST and unconditionally. Watching a replay from the
    // menu creates no CSmArenaClient playground at all, so gating the vehicle
    // read on the race context (as this used to) reported "no car" for every
    // frame of a replay. VehicleState's own docs warn about exactly this:
    // the vis state can be valid even when there is no viewing player.
    auto vis = VehicleState::ViewingPlayerState();

    // Race context is a bonus when a playground exists, not a precondition.
    auto app = cast<CTrackMania>(GetApp());
    auto pg = app is null ? null : cast<CSmArenaClient>(app.CurrentPlayground);
    CGameTerminal@ term = null;
    CSmScriptPlayer@ api = null;
    int ui = 0;
    string mapUid = "";
    string mapName = "";

    if (pg !is null) {
        if (pg.Map !is null && pg.Map.MapInfo !is null) {
            mapUid = pg.Map.MapInfo.MapUid;
            // Human-readable title. Strip the $-format codes and any quote /
            // backslash so it drops straight into the JSON string below.
            mapName = Text::StripFormatCodes(pg.Map.MapInfo.Name);
            mapName = mapName.Replace("\\", " ").Replace("\"", "'");
        }
        if (pg.GameTerminals.Length > 0) {
            @term = pg.GameTerminals[0];
            ui = int(term.UISequence_Current);
            auto player = cast<CSmPlayer>(term.GUIPlayer);
            if (player !is null) @api = cast<CSmScriptPlayer>(player.ScriptAPI);
        }
    }

    string extra = S_AllVehicles ? VehiclesJson() : "";
    // Splitscreen: one entry per local player. Empty on a single-player game,
    // so nothing downstream has to care which mode it is.
    if (S_Players) extra += PlayersJson(pg);

    if (vis is null) {
        // Heartbeat, so the policy can tell "no car yet" from "plugin died".
        //
        // The map uid goes in here too. It is known whenever a playground
        // exists, and in SPLITSCREEN most records are heartbeats - the
        // camera is on one seat at a time, so the viewed vehicle state is
        // frequently null. Leaving the uid out meant clients almost never
        // learned which map was loaded, and everything keyed on it fell back
        // to "default": no occupancy grid for a map that had been dumped,
        // materials written to maps/unknown.materials.json, an empty surface
        // list in the panel, and the wrong tuning config.
        return "{\"t\":" + Time::Now + ",\"car\":false,\"ui\":" + ui
             + ",\"map\":\"" + mapUid + "\",\"map_name\":\"" + mapName + "\""
             + ",\"in_race\":false" + extra + "}";
    }

    // Prefer CSmScriptPlayer where we have it (cleaner rpm/gear/adherence),
    // but fall back to the vis state so replays still produce full samples.
    bool haveApi = api !is null;

    string s = "{\"t\":" + Time::Now
        + ",\"car\":true"
        + ",\"ui\":" + ui
        + ",\"in_race\":" + (haveApi ? "true" : "false")
        + ",\"source\":\"" + (haveApi ? "player" : "viewed") + "\""
        + ",\"map\":\"" + mapUid + "\",\"map_name\":\"" + mapName + "\""
        + ",\"spawn\":" + (haveApi ? int(api.SpawnStatus) : 2)
        + ",\"race_time\":" + (haveApi ? RaceTimeMs(api) : int(vis.RaceStartTime))
        + ",\"finished\":" + (term !is null
             && term.UISequence_Current == CGamePlaygroundUIConfig::EUISequence::Finish
             ? "true" : "false")
        // RaceWaypointTimes reads 0 in Time Attack even after passing
        // checkpoints, which is why SAC_GetData goes via the PlayerState
        // plugin. Emit both candidates so the client can use whichever moves.
        + ",\"cp\":" + (haveApi ? api.RaceWaypointTimes.Length : 0)
        + ",\"cp_lap\":" + (haveApi ? api.CurrentLapWaypointTimes.Length : 0)
        + ",\"respawns\":" + (haveApi ? api.CurrentRaceRespawns : 0)
        + ",\"pos\":" + V3(haveApi ? api.Position : vis.Position)
        + ",\"vel\":" + V3(haveApi ? api.Velocity : vis.WorldVel)
        + ",\"dir\":" + V3(vis.Dir)
        + ",\"up\":" + V3(vis.Up)
        + ",\"left\":" + V3(vis.Left)
        + ",\"speed\":" + F(haveApi ? api.Speed : vis.FrontSpeed)
        + ",\"rpm\":" + F(haveApi ? api.EngineRpm : VehicleState::GetRPM(vis))
        + ",\"gear\":" + (haveApi ? api.EngineCurGear : int(vis.CurGear))
        + ",\"dist\":" + F(haveApi ? api.Distance : 0.0f)
        + ",\"in_steer\":" + F(haveApi ? api.InputSteer : vis.InputSteer)
        + ",\"in_gas\":" + F(haveApi ? api.InputGasPedal : vis.InputGasPedal)
        + ",\"in_brake\":" + ((haveApi ? api.InputIsBraking : vis.InputIsBraking)
                              ? "true" : "false")
        + ",\"adherence\":" + F(haveApi ? api.AdherenceCoef : 1.0f)
        + ",\"skidding\":" + (haveApi ? api.WheelsSkiddingCount : 0)
        + ",\"flying\":" + (haveApi ? api.FlyingDuration : 0)
        + ",\"ground\":" + (vis.IsGroundContact ? "true" : "false")
        + ",\"ground_dist\":" + F(vis.GroundDist)
        + ",\"top_contact\":" + (vis.IsTopContact ? "true" : "false")
        + ",\"turbo\":" + (vis.IsTurbo ? "true" : "false")
        + EffectsJson(vis)
        + (S_Wheels ? WheelsJson(vis) : "")
        + extra
        + "}";
    return s;
}

// ---------------------------------------------------------------------------
// Command channel. The same socket carries telemetry out and commands in, so
// the policy needs only one connection.
//
// Everything here is a real Openplanet API, lifted from how the ManiaExchange
// plugin plays a map - no simulated clicks anywhere:
//   restart          RequestRestartMap()   - the episode reset primitive
//   landmarks        Arena.MapLandmarks    - checkpoint/finish/spawn positions
//   dumpmap [occupancy]                    - effect blocks; + solid grid cells
//   goto <mapuid>    RequestGotoMap()      - switch map by uid, already loaded
//   playmap <url>    PlayMap()             - load an arbitrary map, e.g. a TMX
//                                            /mapgbx/<id> url
//   menu             BackToMainMenu()
//   perms            Permissions::PlayLocalMap() - can this account tier play
//                                            a map off local disk at all?
//   rate <hz>        change the send rate live
//   ping             replies {"pong":true}
// ---------------------------------------------------------------------------

CGamePlaygroundClientScriptAPI@ PlaygroundApi() {
    auto app = cast<CTrackMania>(GetApp());
    if (app is null || app.Network is null) return null;
    return app.Network.PlaygroundClientScriptAPI;
}

// Checkpoint, finish and spawn positions in world space.
//
// This is the honest source for "how many checkpoints have we passed" and for
// "where is the finish on a map nobody has driven". RaceWaypointTimes reads 0
// in Time Attack no matter what, so counting checkpoints from the player API is
// a dead end; the arena's landmark list is static map data and always correct.
// The client counts them by proximity to these positions instead.
string LandmarksJson() {
    auto app = cast<CTrackMania>(GetApp());
    auto pg = app is null ? null : cast<CSmArenaClient>(app.CurrentPlayground);
    if (pg is null || pg.Arena is null) {
        return "{\"ok\":false,\"cmd\":\"landmarks\",\"err\":\"no playground\"}";
    }

    string uid = "";
    if (pg.Map !is null && pg.Map.MapInfo !is null) uid = pg.Map.MapInfo.MapUid;

    auto lms = pg.Arena.MapLandmarks;
    string s = "{\"ok\":true,\"cmd\":\"landmarks\",\"map\":\"" + uid
             + "\",\"count\":" + lms.Length + ",\"items\":[";
    bool first = true;
    for (uint i = 0; i < lms.Length; i++) {
        auto lm = lms[i];
        if (lm is null) continue;

        // Only the three kinds we can act on. A map is full of landmarks that
        // are neither (gauges, sectors, object anchors) and they would just be
        // noise in the client's checkpoint search.
        string kind;
        if (lm.Waypoint !is null) {
            kind = lm.Waypoint.IsFinish ? "finish" : "checkpoint";
        } else if (lm.PlayerSpawn !is null) {
            kind = "spawn";
        } else {
            continue;
        }

        if (!first) s += ",";
        first = false;
        s += "{\"kind\":\"" + kind + "\""
           + ",\"tag\":\"" + lm.Tag + "\""
           + ",\"order\":" + lm.Order
           + ",\"multilap\":" + ((lm.Waypoint !is null && lm.Waypoint.IsMultiLap)
                                 ? "true" : "false")
           + ",\"pos\":" + V3(lm.Position) + "}";
    }
    return s + "]}";
}

// Every LOCAL player, not just the one the camera is on.
//
// This is what splitscreen needs. `GameTerminals` is one entry per local
// player, each carrying its own ControlledPlayer and - importantly - its own
// UISequence_Current, so the reset state machine can be run per car rather
// than for the window as a whole.
//
// Why this matters beyond throughput: a Starter Access account cannot start a
// local map in solo, but local multiplayer can, and the extra players in a
// splitscreen game are guests rather than signed-in accounts. One account
// therefore drives two or four cars, which is the account ceiling divided by
// four rather than a workaround for it.
//
// The per-wheel detail (slip, surface) still comes from the vis states, which
// are matched to players by position: GetAllVis returns the scene's vehicles
// in no particular order and nothing links a vis back to its player directly.
string PlayersJson(CSmArenaClient@ pg) {
    if (pg is null) return "";
    auto terms = pg.GameTerminals;
    if (terms.Length == 0) return "";

    auto scene = GetApp().GameScene;
    array<CSceneVehicleVisState@> vis;
    if (scene !is null) {
        auto all = VehicleState::GetAllVis(scene);
        for (uint i = 0; i < all.Length; i++) {
            if (all[i].AsyncState !is null) vis.InsertLast(all[i].AsyncState);
        }
    }

    string s = ",\"players\":[";
    for (uint i = 0; i < terms.Length; i++) {
        auto term = terms[i];
        if (term is null) continue;
        auto player = cast<CSmPlayer>(term.ControlledPlayer !is null
                                      ? term.ControlledPlayer : term.GUIPlayer);
        CSmScriptPlayer@ api = null;
        if (player !is null) @api = cast<CSmScriptPlayer>(player.ScriptAPI);
        if (i > 0) s += ",";
        if (api is null) {
            s += "{\"slot\":" + i + ",\"car\":false,\"ui\":"
               + int(term.UISequence_Current) + "}";
            continue;
        }

        // Nearest vis to this player's position. On a splitscreen grid the
        // cars are metres apart, so nearest-position is unambiguous; it only
        // becomes tight if two cars are genuinely overlapping, and then the
        // wheel detail is interchangeable anyway.
        CSceneVehicleVisState@ mine = null;
        float best = 9e9f;
        for (uint j = 0; j < vis.Length; j++) {
            auto d = vis[j].Position - api.Position;
            float q = d.x * d.x + d.y * d.y + d.z * d.z;
            if (q < best) { best = q; @mine = vis[j]; }
        }

        s += "{\"slot\":" + i
           + ",\"car\":true"
           + ",\"ui\":" + int(term.UISequence_Current)
           + ",\"race_time\":" + RaceTimeMs(api)
           + ",\"cp\":" + api.RaceWaypointTimes.Length
           + ",\"respawns\":" + api.CurrentRaceRespawns
           + ",\"pos\":" + V3(api.Position)
           + ",\"vel\":" + V3(api.Velocity)
           + ",\"speed\":" + F(api.Speed)
           + ",\"rpm\":" + F(api.EngineRpm)
           + ",\"gear\":" + api.EngineCurGear
           + ",\"dist\":" + F(api.Distance)
           + ",\"in_steer\":" + F(api.InputSteer)
           + ",\"in_gas\":" + F(api.InputGasPedal)
           + ",\"in_brake\":" + (api.InputIsBraking ? "true" : "false")
           + ",\"adherence\":" + F(api.AdherenceCoef);
        if (mine !is null) {
            s += ",\"dir\":" + V3(mine.Dir)
               + ",\"up\":" + V3(mine.Up)
               + ",\"left\":" + V3(mine.Left)
               + ",\"ground\":" + (mine.IsGroundContact ? "true" : "false")
               + ",\"side_speed\":" + F(VehicleState::GetSideSpeed(mine))
               + ",\"slip\":[" + F(mine.FLSlipCoef) + "," + F(mine.FRSlipCoef)
                                + "," + F(mine.RLSlipCoef) + "," + F(mine.RRSlipCoef) + "]"
               + ",\"mat\":[" + int(mine.FLGroundContactMaterial) + ","
                               + int(mine.FRGroundContactMaterial) + ","
                               + int(mine.RLGroundContactMaterial) + ","
                               + int(mine.RRGroundContactMaterial) + "]";
        }
        s += "}";
    }
    return s + "]";
}

// Loading a map has to yield, so it runs as its own coroutine.
void PlayMapCoro(const string &in url) {
    auto app = cast<CTrackMania>(GetApp());
    if (app is null) return;
    print("TMAI Telemetry: loading map " + url);
    // Without this we get stuck on the current map.
    app.BackToMainMenu();
    while (!app.ManiaTitleControlScriptAPI.IsReady) {
        yield();
    }
    // Empty mode string = the map's default (Race/TimeAttack) on TM2020.
    app.ManiaTitleControlScriptAPI.PlayMap(url, "", "");
}

void BackToMenuCoro() {
    auto app = cast<CTrackMania>(GetApp());
    if (app is null) return;
    app.BackToMainMenu();
    while (!app.ManiaTitleControlScriptAPI.IsReady) {
        yield();
    }
}

// ---------------------------------------------------------------------------
// Static map data: where the effect blocks are, and optionally which grid cells
// are solid.
//
// Classification is by block/item model NAME. Hidden Effects can do better for
// items by reading the surface gameplay id out of memory, but that needs
// Dev::GetOffset*, which an unsigned plugin in School mode does not get. The
// name path is Hidden Effects' own fallback and covers blocks, which is where
// almost every booster on a campaign map actually lives.
//
// We deliberately do NOT import from Hidden Effects: it declares no exports,
// so there is nothing to import, and pinning ourselves to another plugin's
// internals would break on their next release.
// ---------------------------------------------------------------------------

string EffectFromName(const string &in modelId) {
    string n = modelId.ToLower();
    if (n.Contains("sign")) return "";

    if (n.Contains("special") || n.Contains("speciaux") || n.Contains("wide")) {
        if (n.Contains("turbo")) {
            if (n.Contains("roulette")) return "turbo_roulette";
            if (n.Contains("turbo2")) return "turbo2";
            return "turbo";
        }
        if (n.Contains("boost")) return n.Contains("boost2") ? "reactor2" : "reactor";
        if (n.Contains("cruise")) return "cruise";
        if (n.Contains("nobrake")) return "no_brakes";
        if (n.Contains("noengine")) return "no_engine";
        if (n.Contains("nosteering")) return "no_steer";
        if (n.Contains("slowmotion")) return "slow_motion";
        if (n.Contains("fragile")) return "fragile";
        if (n.Contains("reset")) return "reset";
    }
    if (n.Contains("gategameplay") || n.Contains("gateexpandablegameplay")) {
        if (n.Contains("stadium")) return "switch_stadium";
        if (n.Contains("snow")) return "switch_snow";
        if (n.Contains("rally")) return "switch_rally";
        if (n.Contains("desert")) return "switch_desert";
    }
    if (n.Contains("nogrip")) return "no_grip";
    if (n.Contains("forceacceleration") || n.Contains("forceaccelerate")
        || n.Contains("forceaccel") || n.Contains("forcedaccel")) return "forced_accel";
    if (n.Contains("bumper")) {
        if (n.Contains("bumper2")) return "bumper2";
        if (n.Contains("barrelroll")) return "bumper_barrelroll";
        return "bumper";
    }
    return "";
}

// A block's occupied grid cells.
//
// CGameCtnBlockUnit.AbsoluteOffset is NOT absolute - measured on a real map it
// returns 0..2 in x/z and 0..1 in y, i.e. an offset *within* the block. Adding
// nothing to it collapsed 2374 cells into 10 distinct ones and left the lidar
// reading "nothing in range" everywhere. So the offsets are sent relative to
// the block's own Coord and combined on the client.
string BlockCellsJson(CGameCtnBlock@ block) {
    auto c = block.Coord;
    string s = "[";
    auto units = block.BlockUnitsE;
    bool first = true;
    for (uint u = 0; u < units.Length; u++) {
        auto unit = units[u];
        if (unit is null) continue;
        auto o = unit.AbsoluteOffset;
        if (!first) s += ",";
        first = false;
        s += "[" + (c.x + o.x) + "," + (c.y + o.y) + "," + (c.z + o.z) + "]";
    }
    if (first) s += "[" + c.x + "," + c.y + "," + c.z + "]";
    return s + "]";
}

// The variant carries the block's footprint in grid cells. Straight out of
// Item Placement Toolbox by way of Hidden Effects - all documented members, no
// Dev:: reads.
CGameCtnBlockInfoVariant@ BlockVariant(CGameCtnBlock@ block) {
    auto info = block.BlockInfo;
    if (info is null) return null;
    uint vi = block.BlockInfoVariantIndex;
    CGameCtnBlockInfoVariant@ v = null;
    if (vi > 0) {
        if (block.IsGround) {
            if (info.AdditionalVariantsGround.Length >= vi)
                @v = cast<CGameCtnBlockInfoVariant>(info.AdditionalVariantsGround[vi - 1]);
        } else {
            if (info.AdditionalVariantsAir.Length >= vi)
                @v = cast<CGameCtnBlockInfoVariant>(info.AdditionalVariantsAir[vi - 1]);
        }
    } else if (block.IsGround) {
        @v = cast<CGameCtnBlockInfoVariant>(info.VariantGround);
        if (v is null) @v = cast<CGameCtnBlockInfoVariant>(info.VariantBaseGround);
    } else {
        @v = cast<CGameCtnBlockInfoVariant>(info.VariantAir);
        if (v is null) @v = cast<CGameCtnBlockInfoVariant>(info.VariantBaseAir);
    }
    return v;
}

string DumpMapJson(bool withOccupancy) {
    auto app = GetApp();
    auto map = app is null ? null : cast<CGameCtnChallenge>(app.RootMap);
    if (map is null) {
        return "{\"ok\":false,\"cmd\":\"dumpmap\",\"err\":\"no map loaded\"}";
    }

    // Grid cell -> world position needs the map's base height. Same vista
    // table Hidden Effects uses; without it every Y is off by a few blocks.
    int baseH = int(map.DecoBaseHeightOffset);
    string vista = map.CollectionName;
    if (vista == "WhiteShore") baseH = 15;
    else if (vista == "GreenCoast") baseH = 5;
    else if (vista == "BlueBay") baseH = 5;
    else if (vista == "RedIsland") baseH = 15;

    string uid = "";
    if (map.MapInfo !is null) uid = map.MapInfo.MapUid;

    auto blocks = map.Blocks;
    auto items = map.AnchoredObjects;
    uint freeSkipped = 0;

    string eff = "";
    bool first = true;
    string occ = "";
    bool firstOcc = true;

    // Distinct block model names, sent once, with each block carrying an index
    // into them. A map has ~2350 blocks but only a few dozen distinct models,
    // so this is a few KB instead of a few hundred - and the client needs the
    // names to tell road from scenery. Without that the grass base plane makes
    // every cell of the map read as solid ground.
    array<string> names;
    dictionary nameIdx;
    string occNames = "";

    for (uint i = 0; i < blocks.Length; i++) {
        auto b = blocks[i];
        if (b is null || b.BlockModel is null) continue;

        // Free blocks store their transform in memory we cannot read without
        // Developer mode. Count them so the client knows coverage is partial
        // rather than silently pretending the map is fully mapped.
        if (int(b.CoordX) < 0) { freeSkipped++; continue; }

        string modelId = b.BlockModel.IdName.Length > 0
            ? b.BlockModel.IdName : string(b.BlockModel.Name);

        if (withOccupancy) {
            // Seven ints per block: coord, cardinal direction, footprint size.
            // The client expands these into cells, because how a footprint
            // rotates about its anchor is the uncertain part and iterating on
            // it here would cost a game reload per guess.
            auto c = b.Coord;
            auto v = BlockVariant(b);
            uint sx = 1, sy = 1, sz = 1;
            if (v !is null) { sx = v.Size.x; sy = v.Size.y; sz = v.Size.z; }

            int ni;
            if (!nameIdx.Get(modelId, ni)) {
                ni = int(names.Length);
                names.InsertLast(modelId);
                nameIdx.Set(modelId, ni);
            }

            if (!firstOcc) occ += ",";
            firstOcc = false;
            occ += c.x + "," + c.y + "," + c.z + "," + int(b.Direction)
                 + "," + sx + "," + sy + "," + sz + "," + ni;
        }

        string type = EffectFromName(modelId);
        if (type.Length == 0) continue;
        if (!first) eff += ",";
        first = false;
        eff += "{\"type\":\"" + type + "\",\"kind\":\"block\",\"cells\":"
             + BlockCellsJson(b) + "}";
    }

    for (uint i = 0; i < items.Length; i++) {
        auto it = items[i];
        if (it is null || it.ItemModel is null) continue;
        string modelId = it.ItemModel.IdName.Length > 0
            ? it.ItemModel.IdName : string(it.ItemModel.Name);
        string type = EffectFromName(modelId);
        if (type.Length == 0) continue;
        if (!first) eff += ",";
        first = false;
        // Items carry a world position already, so no grid maths needed.
        eff += "{\"type\":\"" + type + "\",\"kind\":\"item\",\"pos\":"
             + V3(it.AbsolutePositionInMap) + "}";
    }

    string s = "{\"ok\":true,\"cmd\":\"dumpmap\",\"map\":\"" + uid + "\""
             + ",\"block_size\":[32,8,32],\"base_height\":" + baseH
             + ",\"blocks\":" + blocks.Length + ",\"items\":" + items.Length
             + ",\"free_skipped\":" + freeSkipped
             + ",\"effects\":[" + eff + "]";
    // Flat groups of 8: x, y, z, dir, sx, sy, sz, name index.
    if (withOccupancy) {
        for (uint i = 0; i < names.Length; i++) {
            if (i > 0) occNames += ",";
            occNames += "\"" + names[i] + "\"";
        }
        s += ",\"boxes\":[" + occ + "],\"names\":[" + occNames + "]";
    }
    return s + "}";
}

string HandleCommand(const string &in line) {
    string cmd = line.Trim();
    if (cmd.Length == 0) return "";

    array<string> parts = cmd.Split(" ", 2);
    string verb = parts[0].ToLower();
    string arg = parts.Length > 1 ? parts[1].Trim() : "";

    if (verb == "ping") {
        return "{\"pong\":true}";
    }
    if (verb == "landmarks") {
        return LandmarksJson();
    }
    if (verb == "dumpmap") {
        // Occupancy is opt-in: it is thousands of cells on one line, and only
        // the lidar work in phase 4 needs it.
        return DumpMapJson(arg.ToLower() == "occupancy");
    }
    if (verb == "restart") {
        auto api = PlaygroundApi();
        if (api is null) return "{\"ok\":false,\"cmd\":\"restart\",\"err\":\"no playground\"}";
        api.RequestRestartMap();
        return "{\"ok\":true,\"cmd\":\"restart\"}";
    }
    if (verb == "goto") {
        auto api = PlaygroundApi();
        if (api is null) return "{\"ok\":false,\"cmd\":\"goto\",\"err\":\"no playground\"}";
        if (arg.Length == 0) return "{\"ok\":false,\"cmd\":\"goto\",\"err\":\"need a map uid\"}";
        api.RequestGotoMap(arg);
        return "{\"ok\":true,\"cmd\":\"goto\"}";
    }
    if (verb == "playmap") {
        if (arg.Length == 0) return "{\"ok\":false,\"cmd\":\"playmap\",\"err\":\"need a url\"}";
        startnew(CoroutineFuncUserdataString(PlayMapCoro), arg);
        return "{\"ok\":true,\"cmd\":\"playmap\"}";
    }
    if (verb == "perms") {
        // Can this account actually play a map off the local disk?
        //
        // Worth asking directly rather than inferring from a failure: the
        // symptom of not having this is a map that simply will not start, and
        // the obvious conclusion - "the file must be wrong, rebuild it" - is
        // the expensive wrong answer. A locally rebuilt map is still a local
        // map. If this comes back false, the account tier is the problem and
        // no amount of re-authoring changes it.
        return "{\"ok\":true,\"cmd\":\"perms\""
             + ",\"play_local_map\":" + (Permissions::PlayLocalMap() ? "true" : "false")
             + "}";
    }
    if (verb == "menu") {
        startnew(BackToMenuCoro);
        return "{\"ok\":true,\"cmd\":\"menu\"}";
    }
    if (verb == "rate") {
        S_RateHz = Text::ParseUInt(arg);
        return "{\"ok\":true,\"cmd\":\"rate\",\"hz\":" + S_RateHz + "}";
    }
    return "{\"ok\":false,\"err\":\"unknown command\"}";
}

void Main() {
    print("TMAI Telemetry: starting server on 127.0.0.1:" + S_Port);

    while (true) {
        @g_server = Net::Socket();
        if (!g_server.Listen("127.0.0.1", uint16(S_Port))) {
            print("TMAI Telemetry: listen failed, retrying");
            yield(120);
            continue;
        }

        Net::Socket@ sock = null;
        while (sock is null) {
            yield();
            @sock = g_server.Accept();
        }

        g_clientConnected = true;
        g_lines = 0;
        print("TMAI Telemetry: client connected from " + sock.GetRemoteIP());

        uint lastSend = Time::Now;
        while (!sock.IsHungUp()) {
            if (!sock.IsReady()) { yield(); continue; }

            // Drain any pending commands first. Available() keeps this from
            // blocking the telemetry loop when the policy sends nothing.
            while (sock.Available() > 0) {
                string request;
                if (!sock.ReadLine(request)) break;
                string reply = HandleCommand(request);
                if (reply.Length > 0 && !sock.WriteLine(reply)) break;
            }

            if (S_RateHz > 0) {
                uint period = 1000 / S_RateHz;
                if (Time::Now - lastSend < period) { yield(); continue; }
                lastSend = Time::Now;
            }

            if (!sock.WriteLine(BuildLine())) break;
            g_lines++;
            yield();
        }

        print("TMAI Telemetry: client gone after " + g_lines + " lines");
        sock.Close();
        g_server.Close();
        g_clientConnected = false;
    }
}

void RenderMenu() {
    string status = g_clientConnected ? "\\$0f0client connected" : "\\$f80waiting";
    UI::MenuItem("\\$f0f TMAI Telemetry  " + status, "", false, false);
}
