import traceback
import bittensor as bt
import numpy as np
from bitrecs.utils import epoch
from bitrecs.utils import constants as CONST

def display_normalized_analysis(validator_instance):
    """Display normalized scores that are actually used for weights"""
    try:
        # Calculate the same L1 normalization that set_weights() uses
        raw_scores = validator_instance.scores.copy()
        
        # Apply the same L1 normalization logic from set_weights()
        norm = np.linalg.norm(raw_scores, ord=1, axis=0, keepdims=True)
        
        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN
        
        # Compute normalized scores (same as raw_weights in set_weights)
        normalized_scores = raw_scores / norm
        
        bt.logging.info("\033[1;36m=== NORMALIZED WEIGHTS ===\033[0m")
        
        if len(raw_scores) > 0:
            min_score = np.min(raw_scores)
            max_score = np.max(raw_scores)
            bt.logging.info(f"Raw score range: {min_score:.6f} - {max_score:.6f}")
            if min_score > 0:
                raw_ratio = max_score / min_score
                bt.logging.info(f"Raw score ratio: {raw_ratio:.2f}")
            else:
                bt.logging.info("Raw score ratio: inf (min score is zero)")
            bt.logging.info(f"Active scores: {np.count_nonzero(raw_scores)} (filtered from {len(raw_scores)})")
        else:
            bt.logging.warning("No significant raw scores found")
            return
        
        # Create dictionary for active normalized scores
        active_normalized = {uid: norm_score for uid, norm_score in enumerate(normalized_scores) if norm_score > 1e-10}        
        
        if not active_normalized:
            bt.logging.warning("No active normalized weights")
            return
            
        norm_array = np.array(list(active_normalized.values()))
        norm_stats = {
            'mean': np.mean(norm_array),
            'std': np.std(norm_array),
            'cv': np.std(norm_array) / np.mean(norm_array) if np.mean(norm_array) > 0 else 0,
            'min': np.min(norm_array),
            'max': np.max(norm_array),
            'sum': np.sum(normalized_scores)  # Should be ~1.0 due to L1 norm
        }
        
        bt.logging.info(f"Normalized sum: {norm_stats['sum']:.6f} (should be ~1.0)")
        bt.logging.info(f"Normalized CV: {norm_stats['cv']:.3f}")
        bt.logging.info(f"Normalized range: {norm_stats['min']:.6f} - {norm_stats['max']:.6f}")
        
        # Show top normalized weights
        sorted_normalized = sorted(active_normalized.items(), key=lambda x: x[1], reverse=True)
        bt.logging.info("Top 10 normalized weights:")
        for i, (uid, weight) in enumerate(sorted_normalized[:10], 1):
            percentage = weight * 100
            bt.logging.info(f"  {i}. UID {uid:2d}: {weight:.6f} ({percentage:.2f}%)")
        
        # Show which UIDs were filtered out
        zero_score_uids = [uid for uid, score in enumerate(validator_instance.scores) if score <= 1e-10]
        if zero_score_uids:
            bt.logging.info(f"Zero/near-zero score UIDs: {len(zero_score_uids)}")
            
        # Check for weight concentration
        top_3_weight = sum(weight for _, weight in sorted_normalized[:3])
        if top_3_weight > 0.7:
            bt.logging.warning(f"⚠️High weight concentration in top 3: {top_3_weight:.1%}")
        
    except Exception as e:
        bt.logging.error(f"Error in normalized analysis: {e}")

def display_ema_insights(validator_instance):
    """Show EMA alpha usage patterns"""
    try:
        if not hasattr(validator_instance, 'alpha_history') or len(validator_instance.alpha_history) == 0:
            bt.logging.debug("No alpha history available")
            return
        
        recent_alphas = validator_instance.alpha_history[-10:]  # Last 10 updates
        avg_alpha = np.mean(recent_alphas)
        
        # Count usage patterns
        default_alpha = float(validator_instance.config.neuron.moving_average_alpha)
        #low_alpha_threshold = default_alpha * 0.6  # Assuming low_alpha is ~60% of default
        low_alpha_threshold = default_alpha / 2
        
        low_alpha_usage = sum(1 for a in recent_alphas if a < low_alpha_threshold) / len(recent_alphas)
        high_alpha_usage = sum(1 for a in recent_alphas if a > default_alpha) / len(recent_alphas)
        
        bt.logging.info("\033[1;36m=== EMA INSIGHTS ===\033[0m")
        bt.logging.info(f"Default alpha: {default_alpha:.3f}")
        bt.logging.info(f"Recent avg alpha: {avg_alpha:.3f}")
        bt.logging.info(f"Low alpha usage: {low_alpha_usage:.1%} (failure handling)")
        bt.logging.info(f"High alpha usage: {high_alpha_usage:.1%} (rapid updates)")
        
        # Show alpha trend
        if len(recent_alphas) >= 5:
            alpha_trend = np.polyfit(range(len(recent_alphas)), recent_alphas, 1)[0]
            bt.logging.info(f"Alpha trend: {alpha_trend:+.4f} {'(increasing)' if alpha_trend > 0 else '(decreasing)'}")
        
        # Interpret patterns
        if low_alpha_usage > 0.3:
            bt.logging.warning("⚠️High failure rate detected (frequent low alpha usage)")
        if avg_alpha < default_alpha * 0.8:
            bt.logging.warning("⚠️System in defensive mode (low average alpha)")
        
    except Exception as e:
        bt.logging.error(f"Error in EMA insights: {e}")

def display_transformation_impact(validator_instance):
    """Show impact of non-linear transformation"""
    try:
        if not hasattr(validator_instance, 'scores') or len(validator_instance.scores) == 0:
            return
        
        # Get active scores
        active_scores = validator_instance.scores[validator_instance.scores > 0]
        if len(active_scores) < 2:
            return        
        
        min_threshold = 1e-8
        filtered_scores = active_scores[active_scores > min_threshold]
        
        if len(filtered_scores) < 2:
            bt.logging.warning("⚠️Too few significant scores for transformation analysis")
            return
        
        # Normalize AFTER filtering
        normalized = filtered_scores / np.sum(filtered_scores)
        
        # Apply different powers
        linear = normalized  # Power = 1.0
        moderate = np.power(normalized, 1.2)
        aggressive = np.power(normalized, 1.5)
        
        # Renormalize
        moderate = moderate / np.sum(moderate)
        aggressive = aggressive / np.sum(aggressive)
        
        bt.logging.info("\033[1;36m=== TRANSFORMATION IMPACT ===\033[0m")
        bt.logging.info(f"Scores analyzed: {len(filtered_scores)} (filtered from {len(active_scores)})")
        bt.logging.info(f"Score range: {np.min(filtered_scores):.6f} - {np.max(filtered_scores):.6f}")
        
        # Safe ratio calculation
        linear_nonzero = linear[linear > min_threshold]
        moderate_nonzero = moderate[moderate > min_threshold]
        aggressive_nonzero = aggressive[aggressive > min_threshold]

        linear_ratio = np.max(linear_nonzero) / np.min(linear_nonzero) if len(linear_nonzero) > 1 else float('nan')
        moderate_ratio = np.max(moderate_nonzero) / np.min(moderate_nonzero) if len(moderate_nonzero) > 1 else float('nan')
        aggressive_ratio = np.max(aggressive_nonzero) / np.min(aggressive_nonzero) if len(aggressive_nonzero) > 1 else float('nan')
        
        bt.logging.info(f"Linear (1.0) CV: {np.std(linear)/np.mean(linear):.3f}")
        bt.logging.info(f"Moderate (1.2) CV: {np.std(moderate)/np.mean(moderate):.3f}")
        bt.logging.info(f"Aggressive (1.5) CV: {np.std(aggressive)/np.mean(aggressive):.3f}")
                
        current_power = 1.0
        bt.logging.info(f"Current power: {current_power} ({'linear' if current_power == 1.0 else 'non-linear'})")
        
        # Show amplification effect with safe values
        bt.logging.info(f"Max/Min - Linear: {linear_ratio:.2f}, Moderate: {moderate_ratio:.2f}, Aggressive: {aggressive_ratio:.2f}")
        
        # Warning for extreme ratios
        if linear_ratio > 1000:
            bt.logging.warning("⚠️Score range detected")
        
    except Exception as e:
        bt.logging.error(f"Error in transformation impact: {e}")

def display_score_trends(validator_instance):
    """Display score trends over time"""
    try:
        if not hasattr(validator_instance, 'score_history') or len(validator_instance.score_history) < 2:
            return
            
        current = validator_instance.score_history[-1]
        previous = validator_instance.score_history[-2]
        
        # Calculate changes
        mean_change = current['stats']['mean'] - previous['stats']['mean']
        cv_change = current['stats']['cv'] - previous['stats']['cv']
        
        # Check for leadership changes
        current_leader = current['top_3'][0][0] if current['top_3'] else None
        previous_leader = previous['top_3'][0][0] if previous['top_3'] else None
        
        bt.logging.info("\033[1;35m=== SCORE TRENDS ===\033[0m")
        bt.logging.info(f"Mean change: {mean_change:+.6f}")
        bt.logging.info(f"CV change: {cv_change:+.4f} {'\033[32m(more stable)\033[0m' if cv_change < 0 else '\033[33m(less stable)\033[0m'}")
        
        if current_leader != previous_leader:
            bt.logging.info(f"🏆 Leadership change: UID {previous_leader} → UID {current_leader}")
        
        # Show UIDs that entered/left top 3
        current_top3_uids = {uid for uid, _ in current['top_3']}
        previous_top3_uids = {uid for uid, _ in previous['top_3']}
        
        new_top3 = current_top3_uids - previous_top3_uids
        dropped_top3 = previous_top3_uids - current_top3_uids
        
        if new_top3:
            bt.logging.info(f"📈 Entered top 3: {list(new_top3)}")
        if dropped_top3:
            bt.logging.info(f"📉 Dropped from top 3: {list(dropped_top3)}")
        
        # Show new/lost miners
        new_miners = set(current['active_uids']) - set(previous['active_uids'])
        lost_miners = set(previous['active_uids']) - set(current['active_uids'])
        
        if new_miners:
            bt.logging.info(f"🆕 New active miners: {list(new_miners)}")
        if lost_miners:
            bt.logging.info(f"❌ Lost miners: {list(lost_miners)}")
        
        # Calculate score stability over longer periods
        if len(validator_instance.score_history) >= 5:
            last_5_cvs = [snapshot['stats']['cv'] for snapshot in validator_instance.score_history[-5:]]
            cv_trend = np.polyfit(range(5), last_5_cvs, 1)[0]  # Linear trend
            bt.logging.info(f"CV trend (last 5): {cv_trend:+.4f} {'(stabilizing)' if cv_trend < 0 else '(destabilizing)'}")
    
    except Exception as e:
        bt.logging.error(f"Error in score trends: {e}")

def check_score_health(validator_instance, stats, max_min_ratio):
    """Enhanced health checks based on our convergence work"""
    try:
        health_issues = []
        recommendations = []
        
        # CV-based health (our main convergence metric)
        if stats['cv'] > 0.6:
            health_issues.append("Very high CV - poor convergence")
            recommendations.append("Consider increasing sample_size or alpha")
        elif stats['cv'] > 0.4:
            health_issues.append("High CV - moderate convergence issues")
            recommendations.append("Monitor for improvement, may need tuning")
        elif stats['cv'] < 0.3:
            bt.logging.info(f"✅ Excellent convergence (CV: {stats['cv']:.3f})")
        
        # Score divergence (our Atlantic validator issue)
        if max_min_ratio > 100:
            health_issues.append("Extreme score divergence - geographic/connectivity issues")
            recommendations.append("Check cross-validator consensus")
        elif max_min_ratio > 20:
            health_issues.append("High score divergence")
            recommendations.append("Consider score capping or investigate outliers")
        
        # Network size considerations
        if stats['count'] < 10:
            health_issues.append("Small network - testnet conditions")
            recommendations.append("May need different parameters than production")
        
        # Active miner stability
        if hasattr(validator_instance, 'score_history') and len(validator_instance.score_history) >= 3:
            recent_counts = [s['stats']['count'] for s in validator_instance.score_history[-3:]]
            if max(recent_counts) - min(recent_counts) > 5:
                health_issues.append("Unstable active miner count")
                recommendations.append("Check network connectivity")
        
        # EMA responsiveness check
        if hasattr(validator_instance, 'alpha_history') and len(validator_instance.alpha_history) >= 5:
            recent_alphas = validator_instance.alpha_history[-5:]
            avg_alpha = np.mean(recent_alphas)
            if avg_alpha < 0.3:
                health_issues.append("Low EMA responsiveness")
                recommendations.append("System may be in defensive mode due to failures")
        
        # if health_issues:
        #     bt.logging.warning(f"🩺 Score Health Issues:")
        #     for issue in health_issues:
        #         bt.logging.warning(f"   - {issue}")
            
            # if recommendations:
            #     bt.logging.info(f"💡 Recommendations:")
            #     for rec in recommendations:
            #         bt.logging.info(f"   - {rec}")
        else:
            bt.logging.info("✅ Score health: Good")
    
    except Exception as e:
        bt.logging.error(f"Error in score health check: {e}")

def display_epoch_info(validator_instance):
    """Display epoch information for the current block"""
    try:
        current_block = validator_instance.block
        netuid = validator_instance.config.netuid
        
        current_epoch, blocks_until_next_epoch, epoch_start_block = epoch.get_current_epoch_info(current_block, netuid)

        block_time = 12
        minutes_to_next_block = blocks_until_next_epoch * block_time / 60
        
        bt.logging.info("\033[1;34m=== EPOCH INFO ===\033[0m")
        bt.logging.info(f"NetUID: {netuid}")
        bt.logging.info(f"Epoch start block: {epoch_start_block}")
        bt.logging.info(f"Current block: {current_block}")
        bt.logging.info(f"Current epoch: {current_epoch}")
        bt.logging.info(f"Blocks until next epoch: {blocks_until_next_epoch}")
        bt.logging.info(f"Minutes until next epoch: {minutes_to_next_block:.1f}")
        
    except Exception as e:
        bt.logging.error(f"Error in epoch info display: {e}")

def display_batch_progress(validator_instance):
    """Display progress through the current tempo's batches."""
    try:
        total_batches = len(getattr(validator_instance, "tempo_batches", []))
        #current_index = getattr(validator_instance, "tempo_batch_index", 0)
        if total_batches == 0:
            bt.logging.info("No batches initialized for this tempo.")
            return
        #percent = (current_index / total_batches) * 100
        cyan = "\033[36m"
        bold = "\033[1m"
        reset = "\033[0m"
        bt.logging.info("\033[1;36m=== BATCH PROGRESS ===\033[0m")
        bt.logging.info(f"Total Size: {len(validator_instance.total_uids)}")
        bt.logging.info(f"Batch Size: {CONST.QUERY_BATCH_SIZE}")
        batches_completed = getattr(validator_instance, "batches_completed", 0)
        bt.logging.info(
            f"Processed batches: {cyan}{batches_completed}{reset}/{cyan}{total_batches}{reset} "
            f"({bold}{(batches_completed / total_batches) * 100:.1f}%{reset})"
        )
        # bt.logging.info(
        #     f"Processed batches: {cyan}{current_index}{reset}/{cyan}{total_batches}{reset} "
        #     f"({bold}{percent:.1f}%{reset})"
        # )

        percent_seen = (len(validator_instance.batch_seen_uids) / len(validator_instance.total_uids)) * 100 if len(validator_instance.total_uids) > 0 else 0
        bt.logging.info(
            f"Processed uids: {cyan}{len(validator_instance.batch_seen_uids)}{reset} "
            f"({bold}{percent_seen:.1f}%{reset} of total {len(validator_instance.total_uids)})"
        )
       
    except Exception as e:
        bt.logging.error(f"Error in batch progress display: {e}")

def display_score_histogram(validator_instance, bins=20, width=20):
    """
    Display a compact ASCII histogram of node scores in the terminal/logs.
    """
    scores = np.array(validator_instance.scores)
    if len(scores) == 0:
        bt.logging.trace("No scores to display.")
        return

    hist, bin_edges = np.histogram(scores, bins=bins)
    max_count = np.max(hist)
    bt.logging.trace(f"=== SCORE HISTOGRAM ({bins} bins) ===")
    for i in range(bins):
        center = (bin_edges[i] + bin_edges[i+1]) / 2
        count = hist[i]
        bar = '█' * int(width * count / max_count) if max_count > 0 else ''
        bt.logging.trace(f"{center:5.2f} | {bar:<{width}} {count}")
    bt.logging.trace(f"Min:{scores.min():.4f}, Max:{scores.max():.4f}, Mean:{scores.mean():.4f}, Std:{scores.std():.4f}")


def run_complete_score_analysis(validator_instance):
    """Run all score analysis functions in sequence"""
    try:
        #bt.logging.info(f"\033[1;36m=== ENHANCED SCORE ANALYSIS ===\033[0m")
        display_normalized_analysis(validator_instance)
        #display_ema_insights(validator_instance)
        #display_transformation_impact(validator_instance)
        display_score_trends(validator_instance)
        display_epoch_info(validator_instance)
        display_batch_progress(validator_instance)
        display_score_histogram(validator_instance)
        bt.logging.info("\033[1;36m=== SUMMARY COMPLETE ===\033[0m")
    except Exception as e:
        bt.logging.error(f"Error in complete score analysis: {e}")
        bt.logging.error(traceback.format_exc())

