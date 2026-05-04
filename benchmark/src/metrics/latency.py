def latency_score(start_timestamp: float,
                  end_timestamp: float,
                  max_acceptable_latency: float = 10.0) -> float:
    """
    Lower latency = higher score.
    """

    latency = end_timestamp - start_timestamp

    if latency < 0:
        # End before start - invalid timestamps
        raise ValueError(f"Invalid timestamps: end ({end_timestamp}) before start ({start_timestamp})")
    
    if latency == 0:
        # Instant response - perfect score
        return 1.0
    
    if latency >= max_acceptable_latency:
        # Too slow - worst score
        return 0.0

    return 1.0 - (latency / max_acceptable_latency)