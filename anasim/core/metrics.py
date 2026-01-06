
import numpy as np

def compute_performance_error(measured: np.array, target: np.array) -> np.array:
    """
    Compute Performance Error (PE) = (Measured - Target) / Target * 100.
    """
    pe = np.zeros_like(measured)
    mask = (target != 0)
    pe[mask] = (measured[mask] - target[mask]) / target[mask] * 100.0
    return pe

def compute_mdpe(pe: np.array) -> float:
    return np.median(pe)

def compute_mdape(pe: np.array) -> float:
    return np.median(np.abs(pe))

def compute_wobble(pe: np.array, mdpe: float = None) -> float:
    if mdpe is None:
        mdpe = np.median(pe)
    return np.median(np.abs(pe - mdpe))


def compute_control_metrics(time: list, measured: list, target: list, 
                            start_time: float = 0.0, end_time: float = None) -> dict:
    """
    Compute Varvel metrics for TCI performance.
    
    Args:
        time: List of timestamps (s)
        measured: List of measured values (e.g. BIS)
        target: List of target values
        start_time: Start of evaluation window (s)
        end_time: End of evaluation window (s)
        
    Returns:
        dict: {MDPE, MDAPE, Wobble, GlobalScore}
    """
    t_arr = np.array(time)
    m_arr = np.array(measured)
    tgt_arr = np.array(target)
    
    if end_time is None:
        end_time = t_arr[-1]
        
    mask = (t_arr >= start_time) & (t_arr <= end_time)
    
    if not np.any(mask):
        return {"MDPE": 0, "MDAPE": 0, "Wobble": 0, "GlobalScore": 0}
        
    m_win = m_arr[mask]
    tgt_win = tgt_arr[mask]
    
    pe = compute_performance_error(m_win, tgt_win)
    
    mdpe = compute_mdpe(pe)
    mdape = compute_mdape(pe)
    wobble = compute_wobble(pe, mdpe)

    gs = mdape + wobble
    
    return {
        "MDPE": mdpe,
        "MDAPE": mdape,
        "Wobble": wobble,
        "GlobalScore": gs
    }
