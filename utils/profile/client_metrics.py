import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, MutableMapping, Any, Union, Sequence, List, Mapping
from enum import Enum
import numpy as np

class PreferTime(Enum):
    SLOW = 1      # favorece índices MAIS altos
    UNIFORM = 2   # distribuição uniforme de índices
    QUICK = 3     # favorece índices MAIS baixos

class PreferBattery(Enum):
    EQUAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4

class PreferCarbon(Enum):
    LOW = 1
    UNIFORM = 2
    HIGH = 3


def get_clients_time(
    unique_times: Sequence[Any],
    num_clients: int,
    seed: int,
    profile_pref: Union[str, PreferTime] = "UNIFORM",
) -> List[Any]:
    """
    Gera uma lista de tempos (valores de `unique_times`) de tamanho `num_clients`,
    de acordo com a preferência de tempo.

    profile_pref:
      - 'SLOW'    : distribuição lognormal espelhada (mais valores próximos ao max_idx)
      - 'UNIFORM' : uniforme nos índices
      - 'QUICK'   : distribuição lognormal direta (mais valores próximos ao min_idx)
    """
    if not unique_times:
        raise ValueError("unique_times não pode ser vazio.")
    if num_clients < 0:
        raise ValueError("num_clients não pode ser negativo.")

    rng = np.random.default_rng(seed)
    min_idx = 0
    max_idx = len(unique_times) - 1

    # Normaliza a preferência (aceita Enum ou string)
    if isinstance(profile_pref, PreferTime):
        pref = profile_pref
    else:
        try:
            pref = PreferTime[str(profile_pref).upper()]
        except KeyError as e:
            raise ValueError(f"Preferência desconhecida: {profile_pref!r}") from e

    def scaled_lognormal(size: int) -> np.ndarray:
        """Lognormal reescalonada para o intervalo [min_idx, max_idx]."""
        vals = rng.lognormal(mean=0.0, sigma=1.0, size=size)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax == vmin:
            # Caso extremo: todos iguais -> retorna min_idx
            return np.full(size, float(min_idx))
        scaled = (vals - vmin) / (vmax - vmin)          # [0, 1]
        return scaled * (max_idx - min_idx) + min_idx   # [min_idx, max_idx]

    if pref is PreferTime.SLOW:
        base = scaled_lognormal(num_clients)
        # Espelha para favorecer índices altos
        idx = np.rint(max_idx - (base - min_idx)).astype(int)

    elif pref is PreferTime.UNIFORM:
        # high exclusivo -> usar max_idx + 1 para permitir o último índice
        vals = rng.uniform(low=min_idx, high=max_idx + 1, size=num_clients)
        idx = np.rint(vals).astype(int)

    elif pref is PreferTime.QUICK:
        idx = np.rint(scaled_lognormal(num_clients)).astype(int)

    # Garante que arredondamentos fiquem no intervalo
    idx = np.clip(idx, min_idx, max_idx)

    return [unique_times[i] for i in idx]


def get_clients_battery(
    profiles: MutableMapping[Any, Dict[str, Any]],
    seed: int,
    prefer: Union[str, PreferBattery] = "EQUAL",
    kj_low: float = 6,
    kj_medium: float = 18,
    kj_high: float = 30,
) -> MutableMapping[Any, Dict[str, Any]]:
    """
    Atribui bateria inicial/atual (em joules) aos perfis, de acordo com a preferência.

    Preferências (pesos relativos para LOW/MEDIUM/HIGH):
      - EQUAL  -> [1, 1, 1]
      - LOW    -> [2, 1, 1] (mais clientes com bateria baixa)
      - MEDIUM -> [1, 2, 1]
      - HIGH   -> [1, 1, 2]

    Observações:
      - Modifica `profiles` IN PLACE e o retorna.
      - `kj_*` são valores em kJ; serão convertidos para joules (multiplica por 1000).
      - Determinístico dado o `seed`.
    """
    n = len(profiles)
    if n == 0:
        return profiles  # nada a fazer

    # Normaliza preferência para Enum
    if isinstance(prefer, PreferBattery):
        pref = prefer
    else:
        try:
            pref = PreferBattery[str(prefer).upper()]
        except KeyError as e:
            raise ValueError(f"Preferência desconhecida: {prefer!r}") from e

    rng = np.random.default_rng(seed)

    # Define pesos por preferência
    weights_map = {
        PreferBattery.EQUAL:  (1.0, 1.0, 1.0),
        PreferBattery.LOW:    (2.0, 1.0, 1.0),
        PreferBattery.MEDIUM: (1.0, 2.0, 1.0),
        PreferBattery.HIGH:   (1.0, 1.0, 2.0),
    }
    w_low, w_med, w_high = weights_map[pref]
    total = w_low + w_med + w_high
    probs = np.array([w_low / total, w_med / total, w_high / total], dtype=float)

    # Amostra contagens por categoria que somam n (evita arredondamentos ad hoc)
    num_low, num_med, num_high = rng.multinomial(n, probs)

    # Embaralha cids e fatia por contagens -> O(n)
    cids = np.array(list(profiles.keys()))
    rng.shuffle(cids)

    low_slice = cids[:num_low]
    med_slice = cids[num_low : num_low + num_med]
    high_slice = cids[num_low + num_med : num_low + num_med + num_high]

    # Converte kJ -> J -> mJ
    low_J = float(kj_low) * 1000.0 * 1000.0
    med_J = float(kj_medium) * 1000.0 * 1000.0
    high_J = float(kj_high) * 1000.0 * 1000.0

    # Atribui baterias
    for cid in low_slice:
        prof = profiles[cid].copy()
        prof["current_battery_mJ"] = low_J
        prof["initial_battery_mJ"] = low_J
        profiles[cid] = prof

    for cid in med_slice:
        prof = profiles[cid].copy()
        prof["current_battery_mJ"] = med_J
        prof["initial_battery_mJ"] = med_J
        profiles[cid] = prof

    for cid in high_slice:
        prof = profiles[cid].copy()
        prof["current_battery_mJ"] = high_J
        prof["initial_battery_mJ"] = high_J
        profiles[cid] = prof

    return profiles

def assign_carbon_intensity(
    num_clients: int,
    seed: int,
    region: str,
    carbon_data_path: Union[str, Path],
    prefer_carbon: Union[str, "PreferCarbon"] = "UNIFORM",
) -> List[float]:
    """
    Atribui intensidades de carbono para `num_clients` clientes com base em uma região
    (World, continente ou país) presente no arquivo JSON `carbon_data_path`.

    Preferências:
      - LOW      : favorece valores mais baixos (lognormal reescalonada)
      - UNIFORM  : distribuição uniforme nos índices
      - HIGH     : favorece valores mais altos (lognormal espelhada)

    Regras de região:
      - "World"         -> agrupa valores de todos os continentes/países
      - nome de continente presente em World
      - nome de país presente em algum continente

    Retorna:
      Lista de floats (intensidades) de tamanho `num_clients`.
    """
    if num_clients < 0:
        raise ValueError("num_clients não pode ser negativo.")
    if num_clients == 0:
        return []

    # Carrega dados
    path = Path(carbon_data_path)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido em {path}: {e}") from e

    # Extrai todos os valores numéricos para a região
    all_values = _extract_region_values(data, region)
    if not all_values:
        raise ValueError(f"Nenhum valor encontrado para a região '{region}'.")
    # Mantém apenas numéricos
    all_values = [float(v) for v in all_values if isinstance(v, (int, float))]
    if not all_values:
        raise ValueError(f"Os dados da região '{region}' não contêm valores numéricos.")
    all_values.sort()

    min_idx, max_idx = 0, len(all_values) - 1

    # Normaliza preferência (string ou Enum)
    pref_name = getattr(prefer_carbon, "name", str(prefer_carbon)).upper()
    try:
        # Evita importar/depender do Enum aqui; só comparamos por nome
        if pref_name not in {"LOW", "UNIFORM", "HIGH"}:
            raise KeyError
    except KeyError as e:
        raise ValueError(f"Preferência desconhecida: {prefer_carbon!r}") from e

    rng = np.random.default_rng(seed)

    def scaled_lognormal(size: int) -> np.ndarray:
        """Gera lognormal reescalonada para [min_idx, max_idx]."""
        vals = rng.lognormal(mean=0.0, sigma=1.0, size=size)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax == vmin:
            return np.full(size, float(min_idx))
        scaled = (vals - vmin) / (vmax - vmin)          # [0,1]
        return scaled * (max_idx - min_idx) + min_idx   # [min_idx, max_idx]

    if pref_name == "HIGH":
        base = scaled_lognormal(num_clients)
        idx = np.rint(max_idx - (base - min_idx)).astype(int)
    elif pref_name == "UNIFORM":
        # high é exclusivo para uniform -> usar max_idx+1 para permitir o último índice
        vals = rng.uniform(low=min_idx, high=max_idx + 1, size=num_clients)
        idx = np.rint(vals).astype(int)
    else:  # LOW
        idx = np.rint(scaled_lognormal(num_clients)).astype(int)

    # Garante que os índices estejam no intervalo após o arredondamento
    idx = np.clip(idx, min_idx, max_idx)

    # Mapeia índices para valores
    return [all_values[i] for i in idx]


def _extract_region_values(data: Mapping[str, Any], region: str) -> List[float]:
    """
    Extrai (aplana) os valores numéricos da estrutura esperada:
    data["World"][<continent>][<country>] -> dict de valores (ex.: ano->intensidade)

    Casos:
      - region == "World": valores de todos os países
      - region é continente: valores de todos os países do continente
      - region é país: valores do país (busca em todos os continentes)

    Lança ValueError se a região não existir.
    """
    world = data.get("World")
    if not isinstance(world, Mapping):
        raise ValueError("Estrutura de dados inválida: chave 'World' ausente ou malformada.")

    # World inteiro
    if region == "World":
        vals: List[float] = []
        for continent in world.values():
            if isinstance(continent, Mapping):
                for country in continent.values():
                    if isinstance(country, Mapping):
                        vals.extend(country.values())
        return vals

    # Continente
    if region in world and isinstance(world[region], Mapping):
        vals: List[float] = []
        for country in world[region].values():
            if isinstance(country, Mapping):
                vals.extend(country.values())
        return vals

    # País (buscar em todos os continentes)
    for continent in world.values():
        if isinstance(continent, Mapping) and region in continent:
            country = continent[region]
            if isinstance(country, Mapping):
                return list(country.values())

    raise ValueError(f"Região '{region}' não encontrada nos dados.")


def get_training_times_info(devices):
    # Group devices by inference time
    training_time_to_devices = defaultdict(list)
    for device, specs in devices.items():
        training_time_to_devices[specs["training_ms"]].append(device)

    # Get unique sorted inference times
    unique_training_times = sorted(training_time_to_devices.keys())

    return unique_training_times, training_time_to_devices

# def get_clients_carbon_intensity(profiles, seed, carbon_data_path, carbon_region, prefer_carbon):
#     num_clients = len(profiles.keys())
#     carbon_footprint_per_client = assign_carbon_intensity(num_clients, seed, carbon_region, carbon_data_path,
#                                                           prefer_carbon)
#
#     idx = 0
#     for cid, profile in profiles.items():
#         profile["device_carbon_intensity"] = carbon_footprint_per_client[idx]
#         profiles[cid] = profile
#         idx += 1
#
#     return profiles

def get_clients_carbon_intensity(
    profiles: MutableMapping[Any, Dict[str, Any]],
    seed: int,
    carbon_data_path: Union[str, Path],
    carbon_region: str,
    prefer_carbon: Union[str, "PreferCarbon"] = "UNIFORM",
    *,
    field_name: str = "device_carbon_intensity",
) -> MutableMapping[Any, Dict[str, Any]]:
    """
    Atribui intensidade de carbono a cada perfil em `profiles`, escrevendo em `field_name`.
    Modifica o mapeamento IN PLACE e o retorna.

    - A ordem de atribuição segue a ordem de inserção de `profiles`.
    - `prefer_carbon` pode ser string ou Enum (usa-se .name se houver).
    """
    n = len(profiles)
    if n == 0:
        return profiles

    footprints = assign_carbon_intensity(
        num_clients=n,
        seed=seed,
        region=carbon_region,
        carbon_data_path=carbon_data_path,
        prefer_carbon=prefer_carbon,
    )

    if len(footprints) != n:
        raise RuntimeError(
            f"Número de intensidades ({len(footprints)}) diferente do número de perfis ({n})."
        )

    for (cid, prof), value in zip(profiles.items(), footprints):
        prof[field_name] = float(value)
        profiles[cid] = prof  # mantém compatibilidade caso 'prof' seja uma cópia

    return profiles


# def select_network_speeds(profiles, seed, bandwidth_data_path, net_scenario):
#
#     with open(bandwidth_data_path, "r") as file:
#         network_speeds = json.load(file)
#
#     rng = np.random.default_rng(seed=seed)
#
#     selected_profiles = []
#
#     num_clients = len(profiles.keys())
#
#     # Step 1: Choose between 5G and 4G randomly (equal probability)
#     network_types = rng.choice(["5G", "4G"], size=num_clients)
#
#     for type in network_types:
#         # Step 2: Choose a random index from available speeds
#         index = rng.integers(0, len(network_speeds[net_scenario][type]["up"]) - 1)
#
#         # Step 3: Get the base upload and download speeds
#         up_speed = float(network_speeds[net_scenario][type]["up"][index]) * 1000000 # Mbps -> bps
#         down_speed = float(network_speeds[net_scenario][type]["down"][index]) * 1000000 # Mbps -> bps
#         comm = network_speeds[net_scenario][type]["comm"] # j/bit
#
#         # Store result
#         selected_profiles.append({
#             "up": up_speed,
#             "down": down_speed,
#             "comm": comm
#         })
#
#     idx = 0
#     for cid, profile in profiles.items():
#         profile["up_speed"] = selected_profiles[idx]["up"]
#         profile["down_speed"] = selected_profiles[idx]["down"]
#         profile["comm_joules"] = selected_profiles[idx]["comm"]
#         profiles[cid] = profile
#         idx += 1
#
#     return profiles

def select_network_speeds(
    profiles: MutableMapping[Any, Dict[str, Any]],
    seed: int,
    bandwidth_data_path: Union[str, Path],
    net_scenario: str,
    *,
    type_probs: Mapping[str, float] | None = None,  # ex.: {"5G": 0.7, "4G": 0.3}
) -> MutableMapping[Any, Dict[str, Any]]:
    """
    Para cada cliente, seleciona um tipo de rede (5G/4G) e uma combinação (up/down[, comm])
    do cenário fornecido, atribuindo:
      - profile["up_speed"]     (em bps; origem: Mbps * 1e6)
      - profile["down_speed"]   (em bps; origem: Mbps * 1e6)
      - profile["comm_joules"]  (J/bit; escalar por tipo ou por índice)

    Requer estrutura de dados:
    data[net_scenario][type]["up"]   -> lista de Mbps
    data[net_scenario][type]["down"] -> lista de Mbps
    data[net_scenario][type]["comm"] -> escalar J/bit OU lista J/bit (mesmo tamanho de up/down)

    Modifica `profiles` in-place e retorna o próprio `profiles`.
    """
    n = len(profiles)
    if n == 0:
        return profiles

    # Carrega JSON
    path = Path(bandwidth_data_path)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Arquivo não encontrado: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido em {path}: {e}") from e

    # Valida cenário
    if net_scenario not in data or not isinstance(data[net_scenario], dict):
        raise KeyError(f"Cenário '{net_scenario}' não encontrado no arquivo {path}.")
    scenario = data[net_scenario]

    # Tipos suportados (precisam existir no JSON)
    types_available = [t for t in ("5G", "4G") if t in scenario]
    if not types_available:
        raise KeyError(f"Nenhum dos tipos '5G' ou '4G' encontrado em '{net_scenario}'.")

    # Probabilidades (default: 50/50 nos tipos presentes)
    if type_probs is None:
        p = {t: 1.0 / len(types_available) for t in types_available}
    else:
        # Mantém apenas tipos presentes e renormaliza
        p = {t: float(type_probs.get(t, 0.0)) for t in types_available}
        s = sum(p.values())
        if s <= 0:
            raise ValueError("type_probs inválido: soma das probabilidades deve ser > 0.")
        p = {t: v / s for t, v in p.items()}

    rng = np.random.default_rng(seed)
    # Sorteia tipos respeitando probs
    types_array = rng.choice(types_available, size=n, p=[p[t] for t in types_available])

    # Saídas
    up_out = np.empty(n, dtype=float)
    down_out = np.empty(n, dtype=float)
    comm_out = np.empty(n, dtype=float)

    def _select_for_type(t: str, mask: np.ndarray) -> None:
        """Seleciona índices e preenche saídas para o tipo t nas posições do mask."""
        spec = scenario.get(t)
        if not isinstance(spec, dict):
            raise KeyError(f"Estrutura inválida para o tipo '{t}' no cenário '{net_scenario}'.")

        up_list = spec.get("up")
        down_list = spec.get("down")
        comm = spec.get("comm")

        if not isinstance(up_list, Sequence) or not isinstance(down_list, Sequence):
            raise ValueError(f"Campos 'up' e 'down' devem ser listas para o tipo '{t}'.")
        if len(up_list) == 0 or len(down_list) == 0:
            raise ValueError(f"Listas 'up'/'down' vazias para o tipo '{t}'.")
        if len(up_list) != len(down_list):
            raise ValueError(f"Tamanhos diferentes entre 'up' e 'down' para o tipo '{t}'.")

        k = mask.sum()
        if k == 0:
            return

        # Sorteio de índices (alto exclusivo): usar high=len(...) para permitir o último elemento
        idxs = rng.integers(low=0, high=len(up_list), size=k)

        up_vals = (np.array(up_list, dtype=float)[idxs] * 1_000_000.0)   # Mbps -> bps
        down_vals = (np.array(down_list, dtype=float)[idxs] * 1_000_000.0)

        # comm pode ser escalar ou lista
        if isinstance(comm, Sequence) and not isinstance(comm, (str, bytes)):
            if len(comm) != len(up_list):
                raise ValueError(f"'comm' lista precisa ter o mesmo tamanho de 'up'/'down' para '{t}'.")
            comm_vals = np.array(comm, dtype=float)[idxs]
        else:
            comm_vals = float(comm)

        # Preenche nas posições marcadas
        up_out[mask] = up_vals
        down_out[mask] = down_vals
        if np.isscalar(comm_vals):
            comm_out[mask] = comm_vals
        else:
            comm_out[mask] = comm_vals

    # Aplica por tipo (em lote)
    for t in types_available:
        mask = (types_array == t)
        _select_for_type(t, mask)

    # Atribui aos perfis em ordem determinística
    for i, (cid, prof) in enumerate(profiles.items()):
        prof["up_speed"] = float(up_out[i])
        prof["down_speed"] = float(down_out[i])
        prof["comm_joules"] = float(comm_out[i])
        profiles[cid] = prof

    return profiles

def assign_client_profiles(device_profiles, num_clients, mode="equal", *, allowed_devices=None, speed_key="training_ms",
                           gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    devs = list(device_profiles.keys()) if allowed_devices is None else [d for d in allowed_devices if
                                                                         d in device_profiles]
    if mode == "equal":
        m = len(devs)
        base = num_clients // m
        rem = num_clients - base * m
        counts = {d: base for d in devs}
        if rem > 0:
            extra = rng.choice(devs, size=rem, replace=False if rem <= m else True)
            for d in extra:
                counts[d] += 1
        out = []
        for d, c in counts.items():
            out.extend([device_profiles[d]] * c)
        rng.shuffle(out)
        return out
    else:
        speeds = np.array([max(1e-9, float(device_profiles[d][speed_key])) for d in devs])
        if mode == "fast":
            w = (1.0 / speeds) ** float(gamma)
        else:
            w = (speeds) ** float(gamma)
        w = w / w.sum()
        chosen = rng.choice(devs, size=num_clients, replace=True, p=w).tolist()
        return [device_profiles[d] for d in chosen]