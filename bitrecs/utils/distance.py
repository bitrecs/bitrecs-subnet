import json_repair
from typing import List, Optional, Set
from bitrecs.protocol import BitrecsRequest
from bitrecs.utils.color import ColorScheme, ColorPalette


def calculate_jaccard_distance(set1: Set, set2: Set) -> float:  
    if not set1 or not set2:
        return 1.0        
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 1.0        
    similarity = intersection / union
    distance = 1 - similarity
    return distance


def rec_list_to_set(recs: list) -> Set[str]:
    """
    Convert a list of recommendations to a set of SKUs.
    
    Args:
        recs: List of recommendations (can be dicts or strings)
    Returns:
        Set of SKUs
    """
    sku_set = set()
    for item in recs:
        if isinstance(item, dict) and 'sku' in item:
            sku_set.add(item['sku'])
        elif isinstance(item, str):       
            product = json_repair.loads(item)
            if isinstance(product, dict) and 'sku' in product:
                sku_set.add(product['sku'])
        else:
            print(f"Invalid item type in results: {item}")
    return sku_set


def select_most_similar_sets(rec_sets: List[Set], top_n: int = 2) -> List[int]:
    """
    Select most similar sets based on Jaccard similarity.
    Returns indices of the top N most similar pairs.
    
    Args:
        rec_sets: List of sets to compare
        top_n: Number of indices to return (default 2)
    Returns:
        List of indices for the most similar sets
    """
    n = len(rec_sets)
    all_pairs = []
    
    # Calculate similarities for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            distance = calculate_jaccard_distance(rec_sets[i], rec_sets[j])
            similarity = 1 - distance
            all_pairs.append((similarity, i, j))
    
    # Sort by similarity (highest first)
    all_pairs.sort(reverse=True)
    
    # Debug info
    # print("\nTop similarity pairs:")
    # for sim, i, j in all_pairs[:5]:
    #     print(f"Sets {i},{j}: similarity={sim:.3f} (distance={1-sim:.3f})")
    
    # Get indices from top pairs
    selected = set()
    result = []
    
    # Take indices from best pairs until we have top_n
    for sim, i, j in all_pairs:
        for idx in (i, j):
            if idx not in selected and len(result) < top_n:
                selected.add(idx)
                result.append(idx)
        if len(result) >= top_n:
            break
            
    return result[:top_n]


def select_most_similar_bitrecs(rec_sets: List[BitrecsRequest], top_n: int = 2) -> List[BitrecsRequest]:
    """
    Select most similar BitrecsRequest objects based on their SKU recommendations.
    
    Args:
        rec_sets: List of BitrecsRequest objects
        top_n: Number of similar sets to return
    Returns:
        List of most similar BitrecsRequest objects
    """
    if len(rec_sets) < 2:
        return rec_sets
    
    sku_sets = []
    for req in rec_sets:
        this_set = rec_list_to_set(req.results)
        if this_set:
            sku_sets.append(this_set)
    if not sku_sets:
        print("No valid SKUs found in results")
        return []
    sim = select_most_similar_sets(sku_sets, top_n)
    return [rec_sets[i] for i in sim]


# def select_most_similar_sets_with_scores(rec_sets: List[Set], top_n: int = 2) -> List[tuple]:   
#     n = len(rec_sets)
#     all_pairs = []
#     # Calculate similarities for all pairs
#     for i in range(n):
#         for j in range(i + 1, n):
#             distance = calculate_jaccard_distance(rec_sets[i], rec_sets[j])
#             similarity = 1 - distance
#             all_pairs.append((similarity, i, j))

#     # Sort by similarity (highest first)
#     all_pairs.sort(reverse=True)
#     selected = set()
#     result = []
#     # Take indices and similarity from best pairs until we have top_n
#     for sim, i, j in all_pairs:
#         for idx in (i, j):
#             if idx not in selected and len(result) < top_n:
#                 selected.add(idx)
#                 result.append((idx, sim))
#         if len(result) >= top_n:
#             break
#     return result[:top_n]


def select_most_similar_bitrecs_threshold(rec_sets: List[BitrecsRequest], top_n: int = 2, 
                                          similarity_threshold: float = 0.51) -> List[BitrecsRequest]:
    """
    Self-contained function to select most similar BitrecsRequest objects.
    Includes internal Jaccard calculation and similarity checks.
    
    Args:
        rec_sets: List of BitrecsRequest objects
        top_n: Number of similar sets to return (default 2)
        similarity_threshold: Minimum similarity required (default 0.51)
    Returns:
        List of most similar BitrecsRequest objects meeting threshold
    """
    if len(rec_sets) < 2:
        return rec_sets

    def calc_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    # Convert BitrecsRequests to sets of SKUs
    sku_sets = []
    for req in rec_sets:
        sku_set = set(r['sku'] for r in req.results)
        sku_sets.append((sku_set, req))  # Keep original request paired with its SKUs

    # Calculate all pairwise similarities
    pairs = []
    for i in range(len(sku_sets)):
        for j in range(i + 1, len(sku_sets)):
            similarity = calc_jaccard_similarity(sku_sets[i][0], sku_sets[j][0])
            if similarity >= similarity_threshold:
                pairs.append((i, j, similarity))

    # Sort pairs by similarity (highest first)
    pairs.sort(key=lambda x: x[2], reverse=True)

    if not pairs:
        print(f"No pairs found meeting threshold {similarity_threshold}")
        return []

    # Select best pairs meeting criteria
    selected = set()
    selected_requests = []
    
    for i, j, sim in pairs:
        # Add both requests from the pair if we haven't hit top_n
        if len(selected_requests) < top_n:
            if i not in selected:
                selected.add(i)
                selected_requests.append(rec_sets[i])
            if len(selected_requests) < top_n and j not in selected:
                selected.add(j)
                selected_requests.append(rec_sets[j])

    # Print similarity analysis
    print("\nSimilarity Analysis:")
    print(f"Found {len(selected_requests)} sets meeting threshold {similarity_threshold}")
    for idx, req in enumerate(selected_requests):
        model = req.models_used[0] if req.models_used else "unknown"
        if idx < len(pairs):
            print(f" Set {idx}: Model {model} (similarity: {pairs[idx][2]:.3f})")
        else:
            print(f" Set {idx}: Model {model}")

    return selected_requests[:top_n]


def select_most_similar_bitrecs_threshold2(
    rec_sets: List[BitrecsRequest], 
    top_n: int = 2, 
    similarity_threshold: float = 0.51
) -> Optional[List[BitrecsRequest]]:
    """
    Select most similar BitrecsRequest objects meeting similarity threshold.
    Returns None if no pairs meet threshold.
    
    Args:
        rec_sets: List of BitrecsRequest objects
        top_n: Number of similar sets to return
        similarity_threshold: Minimum similarity required
    Returns:
        List of similar BitrecsRequest objects or None if no matches
    """
    if len(rec_sets) < 2:
        return None
        
    # Calculate similarities between all pairs
    similar_pairs = []
    for i in range(len(rec_sets)):
        set1 = set(r['sku'] for r in rec_sets[i].results)
        for j in range(i + 1, len(rec_sets)):
            set2 = set(r['sku'] for r in rec_sets[j].results)
            
            # Calculate Jaccard similarity
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity >= similarity_threshold:
                similar_pairs.append((i, j, similarity))
    
    if not similar_pairs:
        print(f"No pairs found above threshold {similarity_threshold}")
        return None
        
    # Sort by similarity and get best pairs
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    selected = set()
    result = []
    
    # Take best pairs until we have top_n requests
    for i, j, sim in similar_pairs:
        if len(result) >= top_n:
            break
        if i not in selected:
            selected.add(i)
            result.append(rec_sets[i])
        if len(result) < top_n and j not in selected:
            selected.add(j)
            result.append(rec_sets[j])
            
    return result if result else None



def display_rec_matrix(
    rec_sets: List[Set[str]], 
    models_used: List[str], 
    highlight_indices: List[int] = None,
    color_scheme: ColorScheme = ColorScheme.VIRIDIS
) -> str:
    """
    Displays the similarity matrix for recommendation sets.
    Each cell represents the Jaccard distance between two sets.
    Cells are color-coded based on distance.
    
    Args:
        rec_sets: List of recommendation sets
        models_used: List of model names
        highlight_indices: Indices of sets to highlight
        color_scheme: Color scheme to use for visualization
    Returns:
        str: Complete formatted matrix report
    """
    output = []
    colors = ColorPalette.SCHEMES[color_scheme]
    
    output.append(f"\nDistance Matrix - {len(rec_sets)} sets\n")
    
    # Generate header with highlighting
    header = "       "
    for j in range(len(rec_sets)):
        col_num = f"{j:7d}"
        if highlight_indices and j in highlight_indices:
            header += f"{colors['highlight']}{col_num}\033[0m"
        else:
            header += col_num
    output.append(header)
    
    # Track matches for summary
    match_info = []
    
    # Generate matrix with color scheme
    for i in range(len(rec_sets)):
        if highlight_indices and i in highlight_indices:
            row_start = f"{colors['highlight']}{i:4d}  \033[0m"
        else:
            row_start = f"{i:4d}  "
        
        row = []
        for j in range(len(rec_sets)):
            if j < i:
                distance = calculate_jaccard_distance(rec_sets[i], rec_sets[j])
                cell = f"{distance:7.3f}"
                
                if distance < 0.91:
                    match_info.append((i, j, distance, models_used[i], models_used[j]))
                
                if distance < 1.0:
                    if highlight_indices and i in highlight_indices and j in highlight_indices:
                        cell = f"{colors['highlight']}{cell}\033[0m"
                    elif distance <= 0.5:
                        cell = f"{colors['strong']}{cell}\033[0m"
                    elif distance <= 0.7:
                        cell = f"{colors['medium']}{cell}\033[0m"
                    elif distance <= 0.9:
                        cell = f"{colors['weak']}{cell}\033[0m"
                    else:
                        cell = f"{colors['minimal']}{cell}\033[0m"
                row.append(cell)
            else:
                row.append("      -")
        
        output.append(row_start + "".join(row))    
    
    if match_info:
        output.append("-" * 60)
        output.append("\nModel Matches:")
        output.append("-" * 60)
        for i, j, dist, model1, model2 in sorted(match_info, key=lambda x: (1 - x[2]), reverse=True):
            similarity = 1 - dist
            if similarity >= 0.1:
                if similarity >= 0.5:
                    color = colors['strong']
                elif similarity >= 0.3:
                    color = colors['medium']
                elif similarity >= 0.1:
                    color = colors['weak']
                
                output.append(f"{color}Similarity: {similarity:.4f}\033[0m")
                output.append(f"  Model {i}: {model1}")
                output.append(f"  Model {j}: {model2}")
                output.append(f"  Distance: {dist:.4f}")
                output.append("-" * 40)
                if "random" in model1 or "random" in model2:
                    output.append("\033[33m  ⚠️ Warning: Includes random set!\033[0m")
    
    # Add scheme-specific legend
    #output.append(f"\nLegend ({color_scheme.value}):")
    output.append(f"{colors['highlight']}Highlighted Rows/Cols\033[0m: Selected sets")
    output.append(f"{colors['strong']}>= 0.5\033[0m "
                 f"{colors['medium']}>= 0.3\033[0m "
                 f"{colors['weak']}>= 0.1\033[0m "
                 f"{colors['minimal']}> 0.0\033[0m: Match strength")
    
    # output.append("\nNote: Lower distances between sets (real) vs (random)")
    # output.append("      indicate better recommendation quality")
    output.append("=" * 40)
    
    return "\n".join(output)


def display_rec_matrix_html(
    rec_sets: List[Set[str]], 
    models_used: List[str], 
    highlight_indices: List[int] = None
) -> str:
    """
    Generate HTML visualization of the similarity matrix.
    
    Args:
        rec_sets: List of recommendation sets
        models_used: List of model names
        highlight_indices: Indices of sets to highlight
    Returns:
        str: HTML formatted matrix with styling
    """
    
    css = """
    <style>
        .matrix-table { border-collapse: collapse; font-family: monospace; }
        .matrix-table th, .matrix-table td { 
            padding: 6px; 
            border: 1px solid #ddd;
            text-align: right;
        }
        .matrix-header { background-color: #f5f5f5; }
        .highlight { background-color: #fff3b0; }
        .strong { background-color: #90ee90; }
        .medium { background-color: #87ceeb; }
        .weak { background-color: #b19cd9; }
        .minimal { background-color: #e6e6fa; }
        .match-info { margin-top: 20px; }
        .legend { margin-top: 20px; font-size: 0.9em; }
        .warning { color: #ff6b6b; }
    </style>
    """
    
    html = [css, f"<h3>Distance Matrix - {len(rec_sets)} sets</h3>"]
    html.append('<table class="matrix-table">')
    
    # Header row
    header = ['<tr><th class="matrix-header"></th>']
    for j in range(len(rec_sets)):
        cls = 'highlight' if highlight_indices and j in highlight_indices else 'matrix-header'
        header.append(f'<th class="{cls}">{j}</th>')
    header.append('</tr>')
    html.append(''.join(header))
    
    # Matrix rows
    match_info = []
    for i in range(len(rec_sets)):
        row_cls = 'highlight' if highlight_indices and i in highlight_indices else ''
        row = [f'<tr><td class="{row_cls}">{i}</td>']
        
        for j in range(len(rec_sets)):
            if j < i:
                distance = calculate_jaccard_distance(rec_sets[i], rec_sets[j])
                
                if distance < 0.91:
                    match_info.append((i, j, distance, models_used[i], models_used[j]))
                
                # Determine cell class based on distance
                if highlight_indices and i in highlight_indices and j in highlight_indices:
                    cell_cls = 'highlight'
                elif distance <= 0.5:
                    cell_cls = 'strong'
                elif distance <= 0.7:
                    cell_cls = 'medium'
                elif distance <= 0.9:
                    cell_cls = 'weak'
                elif distance < 1.0:
                    cell_cls = 'minimal'
                else:
                    cell_cls = ''
                    
                cell = f'<td class="{cell_cls}">{distance:.3f}</td>'
            else:
                cell = '<td>-</td>'
            row.append(cell)
            
        row.append('</tr>')
        html.append(''.join(row))
    
    html.append('</table>')

    if 1==1:
        html.append('<div class="sku-sets" style="margin-top: 20px; font-family: monospace;">')
        html.append('<h4>SKU Sets:</h4>')
        # Create list of tuples with index, skus, model and sort by highlight status
        sets_to_display = [(i, skus, model) for i, (skus, model) in enumerate(zip(rec_sets, models_used))]
        sets_to_display.sort(key=lambda x: 0 if highlight_indices and x[0] in highlight_indices else 1)
        
        for i, skus, model in sets_to_display:
            cls = 'highlight' if highlight_indices and i in highlight_indices else ''
            html.append(f'<div class="{cls}" style="margin-bottom: 10px;">')
            html.append(f'<p><strong>Set {i}</strong> (Model: {model})</p>')
            html.append('<p style="margin-left: 20px;">')
            html.append(', '.join(skus))  # No sorting, maintain original order
            html.append('</p></div>')
        html.append('</div>')

    
    # Add match information
    if match_info:
        html.append('<div class="match-info">')
        html.append('<h4>Model Matches:</h4>')
        
        for i, j, dist, model1, model2 in sorted(match_info, key=lambda x: (1 - x[2]), reverse=True):
            similarity = 1 - dist
            if similarity >= 0.1:
                cls = 'strong' if similarity >= 0.5 else 'medium' if similarity >= 0.3 else 'weak'
                html.append(f'<div class="{cls}">')
                html.append(f'<p>Similarity: {similarity:.4f}</p>')
                html.append(f'<p>Model {i}: {model1}</p>')
                html.append(f'<p>Model {j}: {model2}</p>')
                html.append(f'<p>Distance: {dist:.4f}</p>')
                if "random" in model1 or "random" in model2:
                    html.append('<p class="warning">Warning: Includes random set!</p>')
                html.append('</div>')
                html.append('<hr>')
        
        html.append('</div>')
    
    # Add legend
    html.append('''
    <div class="legend">
        <h4>Legend:</h4>
        <p><span class="highlight">Highlighted Sets</span></p>
        <p><span class="strong">Strong Match (>= 0.5)</span></p>
        <p><span class="medium">Medium Match (>= 0.3)</span></p>
        <p><span class="weak">Weak Match (>= 0.1)</span></p>
        <p><span class="minimal">Minimal Match (> 0.0)</span></p>
        <p>Note: Lower distances between sets indicate better recommendation quality</p>
    </div>
    ''')
    
    return '\n'.join(html)


