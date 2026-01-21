#!/bin/bash
# Demo script for Research Agent
# Runs example research queries to demonstrate functionality

set -e

API_URL="${API_URL:-http://localhost:8000}"
TIMEOUT=300

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_health() {
    print_header "Checking API Health"

    if curl -s "${API_URL}/health" | grep -q "healthy"; then
        print_success "API is healthy"
    else
        print_error "API is not responding. Start with: ./scripts/run_local.sh"
        exit 1
    fi

    echo ""
    echo "Readiness check:"
    curl -s "${API_URL}/ready" | python3 -m json.tool 2>/dev/null || curl -s "${API_URL}/ready"
}

run_query() {
    local query="$1"
    local domains="$2"
    local description="$3"

    print_header "$description"
    echo "Query: $query"
    echo "Domains: $domains"
    echo ""

    local response
    response=$(curl -s -X POST "${API_URL}/research" \
        -H "Content-Type: application/json" \
        --max-time $TIMEOUT \
        -d "{
            \"query\": \"$query\",
            \"domains\": $domains,
            \"max_iterations\": 2
        }" 2>&1)

    if echo "$response" | grep -q '"status"'; then
        echo "Response:"
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

        local status
        status=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")

        if [ "$status" = "completed" ]; then
            print_success "Research completed successfully"

            # Extract and display key metrics
            echo ""
            echo "Metrics:"
            echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  - Cost: \${data.get('cost_usd', 0):.4f}\")
print(f\"  - Tokens: {data.get('tokens_used', 0)}\")
print(f\"  - Quality Score: {data.get('quality_score', 'N/A')}\")
print(f\"  - Sections: {len(data.get('sections', {}))}\")
print(f\"  - Citations: {len(data.get('citations', []))}\")
" 2>/dev/null || true
        else
            print_warning "Research status: $status"
        fi
    else
        print_error "Request failed"
        echo "$response"
    fi
}

show_metrics() {
    print_header "Current Metrics"
    curl -s "${API_URL}/metrics/json" | python3 -m json.tool 2>/dev/null || curl -s "${API_URL}/metrics/json"
}

show_workflow_graph() {
    print_header "Workflow Graph"
    curl -s "${API_URL}/research/graph" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('ASCII Diagram:')
print(data.get('ascii_diagram', 'N/A'))
print()
print('Node Descriptions:')
for node, desc in data.get('node_descriptions', {}).items():
    print(f'  - {node}: {desc}')
" 2>/dev/null || curl -s "${API_URL}/research/graph"
}

# Main execution
main() {
    echo ""
    echo "Research Agent Demo"
    echo "==================="
    echo ""
    echo "API URL: $API_URL"

    check_health

    show_workflow_graph

    # Example 1: AI/ML query
    run_query \
        "What are the key differences between transformer attention mechanisms and state space models for sequence modeling?" \
        '["ai_ml"]' \
        "Example 1: AI/ML Research"

    # Example 2: Quantum Physics query
    run_query \
        "Explain the current state of quantum error correction and fault-tolerant quantum computing" \
        '["quantum_physics"]' \
        "Example 2: Quantum Physics Research"

    # Example 3: General research query
    run_query \
        "How do large language models handle long context windows and what are the memory limitations?" \
        '["ai_ml", "general"]' \
        "Example 3: Cross-domain Research"

    show_metrics

    print_header "Demo Complete"
    echo "Generated papers are saved in: outputs/"
    echo ""
    echo "To run a custom query:"
    echo "  curl -X POST ${API_URL}/research \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"query\": \"Your research question\", \"domains\": [\"ai_ml\"]}'"
}

# Parse arguments
case "${1:-}" in
    --health)
        check_health
        ;;
    --metrics)
        show_metrics
        ;;
    --graph)
        show_workflow_graph
        ;;
    --help|-h)
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  --health    Check API health only"
        echo "  --metrics   Show current metrics"
        echo "  --graph     Show workflow graph"
        echo "  --help      Show this help message"
        echo ""
        echo "Environment variables:"
        echo "  API_URL     API base URL (default: http://localhost:8000)"
        ;;
    *)
        main
        ;;
esac
