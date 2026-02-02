"""
Example: Distributed parallel optimization using Ray.

This example demonstrates how to use the RayParallelOptimizer to run
multiple independent optimization trajectories in parallel, exploring
different starting positions to find the global optimum.

Key Features Demonstrated:
1. Multi-worker parallel execution
2. GPU resource management
3. Diverse initial position generation
4. Result aggregation and winner selection
5. Performance analysis and visualization

Usage:
    python examples/ray_parallel_example.py
"""

import numpy as np
import ray
from reflector_position.optimizers import (
    RayParallelOptimizer,
    generate_random_initial_positions,
)


def example_basic_parallel_optimization():
    """
    Basic example: Run 8 parallel gradient descent optimizations.
    
    This demonstrates the simplest use case - parallel execution of
    gradient descent with different starting positions.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Parallel Optimization")
    print("="*80)
    
    # Configuration
    NUM_WORKERS = 8
    GPU_FRACTION = 0.25  # 4 workers per GPU
    
    # Scene configuration
    scene_config = {
        "xml_path": "l_shape_scene.xml",
        "reflector_name": "reflector",
    }
    
    # Search space bounds
    bounds = {
        "x_min": 0.0,
        "x_max": 20.0,
        "y_min": 0.0,
        "y_max": 20.0,
    }
    
    # Generate diverse starting positions
    initial_positions = generate_random_initial_positions(
        num_positions=NUM_WORKERS,
        bounds=bounds,
        fixed_z=3.8,
        seed=42,  # For reproducibility
    )
    
    print(f"\nGenerated {NUM_WORKERS} initial positions:")
    for i, pos in enumerate(initial_positions):
        print(f"  Worker {i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    
    # Initialize parallel optimizer
    parallel_optimizer = RayParallelOptimizer(
        num_workers=NUM_WORKERS,
        gpu_fraction=GPU_FRACTION,
        optimizer_method="gradient_descent",
    )
    
    # Optimization parameters (passed to each worker's optimize())
    optimization_params = {
        "num_iterations": 30,
        "learning_rate": 0.5,
        "samples_per_tx": 500_000,
        "max_depth": 13,
        "use_soft_min": True,
        "temperature": 0.2,
    }
    
    # Run parallel optimization
    results = parallel_optimizer.optimize(
        scene_config=scene_config,
        initial_positions=initial_positions,
        optimization_params=optimization_params,
        verbose=True,
    )
    
    # Analyze results
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    best = results["best_result"]
    stats = results["aggregate_stats"]
    
    print(f"\nBest Configuration (Worker #{best['worker_id']}):")
    print(f"  Final Position: [{best['best_position'][0]:.2f}, "
          f"{best['best_position'][1]:.2f}, {best['best_position'][2]:.2f}]")
    print(f"  Final Metric: {best['best_metric']:.4f}")
    print(f"  Time: {best['time_elapsed']:.2f}s")
    
    print(f"\nPerformance:")
    print(f"  Parallel Speedup: {stats['speedup']:.2f}x")
    print(f"  Wall-clock Time: {stats['total_wall_clock_time']:.2f}s")
    print(f"  Avg Worker Time: {stats['mean_time_per_worker']:.2f}s")
    
    # Plot results
    parallel_optimizer.plot_results(results, metric_name="Min RSS (dBm)")
    
    # Cleanup
    parallel_optimizer.shutdown()
    
    return results


def example_grid_search_parallel():
    """
    Example: Parallel grid search with different grid regions.
    
    This demonstrates how to use Ray with grid search, where each
    worker explores a different region of the space.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Parallel Grid Search")
    print("="*80)
    
    NUM_WORKERS = 4
    
    # Divide the space into 4 quadrants
    full_bounds = {"x_min": 0.0, "x_max": 20.0, "y_min": 0.0, "y_max": 20.0}
    
    # Each worker searches a different quadrant
    worker_configs = []
    quadrant_bounds = [
        {"x_min": 0.0, "x_max": 10.0, "y_min": 0.0, "y_max": 10.0},   # Quadrant 1
        {"x_min": 10.0, "x_max": 20.0, "y_min": 0.0, "y_max": 10.0},  # Quadrant 2
        {"x_min": 0.0, "x_max": 10.0, "y_min": 10.0, "y_max": 20.0},  # Quadrant 3
        {"x_min": 10.0, "x_max": 20.0, "y_min": 10.0, "y_max": 20.0}, # Quadrant 4
    ]
    
    # Generate configs for each worker
    for i, bounds in enumerate(quadrant_bounds):
        worker_configs.append({
            "search_bounds": bounds,
            "grid_resolution": 2.0,
        })
    
    print(f"\nDividing space into {NUM_WORKERS} quadrants:")
    for i, config in enumerate(worker_configs):
        bounds = config["search_bounds"]
        print(f"  Worker {i}: x=[{bounds['x_min']:.0f}, {bounds['x_max']:.0f}], "
              f"y=[{bounds['y_min']:.0f}, {bounds['y_max']:.0f}]")
    
    # Scene config
    scene_config = {"xml_path": "l_shape_scene.xml"}
    
    # Initialize parallel optimizer
    parallel_optimizer = RayParallelOptimizer(
        num_workers=NUM_WORKERS,
        gpu_fraction=0.25,
        optimizer_method="grid_search",
    )
    
    # Optimization parameters
    optimization_params = {
        "samples_per_tx": 1_000_000,
        "max_depth": 13,
    }
    
    # Run parallel grid search
    # Note: For grid search, we don't need initial_positions,
    # but we provide dummy ones for interface consistency
    dummy_positions = [np.array([10.0, 10.0, 3.8])] * NUM_WORKERS
    
    results = parallel_optimizer.optimize(
        scene_config=scene_config,
        initial_positions=dummy_positions,
        optimization_params=optimization_params,
        optimizer_configs=worker_configs,
        verbose=True,
    )
    
    # Plot results
    parallel_optimizer.plot_results(results, metric_name="Min RSS (dBm)")
    
    # Cleanup
    parallel_optimizer.shutdown()
    
    return results


def example_hyperparameter_search():
    """
    Example: Parallel hyperparameter search.
    
    This demonstrates using Ray to test different hyperparameter
    combinations in parallel (e.g., different learning rates).
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Parallel Hyperparameter Search")
    print("="*80)
    
    NUM_WORKERS = 6
    
    # Same starting position, different hyperparameters
    base_position = np.array([10.0, 10.0, 3.8])
    initial_positions = [base_position.copy() for _ in range(NUM_WORKERS)]
    
    # Test different learning rates
    learning_rates = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
    
    print(f"\nTesting {NUM_WORKERS} learning rates:")
    for i, lr in enumerate(learning_rates):
        print(f"  Worker {i}: learning_rate = {lr}")
    
    # Scene config
    scene_config = {"xml_path": "l_shape_scene.xml"}
    
    # Initialize parallel optimizer
    parallel_optimizer = RayParallelOptimizer(
        num_workers=NUM_WORKERS,
        gpu_fraction=0.25,
        optimizer_method="gradient_descent",
    )
    
    # Create different optimization params for each worker
    # We'll use a workaround: modify learning_rate in optimizer_config
    optimizer_configs = [
        {"initial_position": base_position[:2]} for _ in range(NUM_WORKERS)
    ]
    
    # Base optimization parameters
    base_opt_params = {
        "num_iterations": 30,
        "samples_per_tx": 500_000,
        "max_depth": 13,
        "use_soft_min": True,
        "temperature": 0.2,
    }
    
    # Run parallel optimization with different learning rates
    # Note: This is a simplified example. For true hyperparameter tuning,
    # consider using Ray Tune instead.
    all_results = []
    
    for i, lr in enumerate(learning_rates):
        opt_params = base_opt_params.copy()
        opt_params["learning_rate"] = lr
        
        # Run single worker (for demonstration)
        single_worker_optimizer = RayParallelOptimizer(
            num_workers=1,
            gpu_fraction=0.5,
            optimizer_method="gradient_descent",
        )
        
        result = single_worker_optimizer.optimize(
            scene_config=scene_config,
            initial_positions=[initial_positions[i]],
            optimization_params=opt_params,
            verbose=False,
        )
        
        all_results.append({
            "learning_rate": lr,
            "result": result["best_result"],
        })
        
        single_worker_optimizer.shutdown()
    
    # Analyze hyperparameter search results
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*80)
    print(f"\n{'LR':<8} {'Metric':<12} {'Time (s)':<10}")
    print("-" * 30)
    
    best_lr = None
    best_metric = -float('inf')
    
    for res in all_results:
        lr = res["learning_rate"]
        metric = res["result"]["best_metric"]
        time_elapsed = res["result"]["time_elapsed"]
        
        print(f"{lr:<8.2f} {metric:<12.4f} {time_elapsed:<10.2f}")
        
        if metric > best_metric:
            best_metric = metric
            best_lr = lr
    
    print("-" * 30)
    print(f"\nBest learning rate: {best_lr}")
    print(f"Best metric: {best_metric:.4f}")
    
    return all_results


def example_production_workflow():
    """
    Example: Production-grade parallel optimization workflow.
    
    This demonstrates best practices for production use:
    1. Proper error handling
    2. Checkpointing intermediate results
    3. Adaptive resource allocation
    4. Result persistence
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Production Workflow")
    print("="*80)
    
    import os
    import json
    from datetime import datetime
    
    # Configuration
    NUM_WORKERS = 16
    GPU_FRACTION = 0.1  # 10 workers per GPU
    OUTPUT_DIR = "results/ray_parallel"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\nOutput directory: {run_dir}")
    
    # Scene configuration
    scene_config = {
        "xml_path": "l_shape_scene.xml",
        "reflector_name": "reflector",
    }
    
    # Search bounds
    bounds = {
        "x_min": 0.0,
        "x_max": 20.0,
        "y_min": 0.0,
        "y_max": 20.0,
    }
    
    # Generate initial positions with good coverage
    initial_positions = generate_random_initial_positions(
        num_positions=NUM_WORKERS,
        bounds=bounds,
        fixed_z=3.8,
        seed=42,
    )
    
    # Save initial positions
    np.save(
        os.path.join(run_dir, "initial_positions.npy"),
        np.array(initial_positions)
    )
    
    # Initialize parallel optimizer
    parallel_optimizer = RayParallelOptimizer(
        num_workers=NUM_WORKERS,
        gpu_fraction=GPU_FRACTION,
        optimizer_method="gradient_descent",
    )
    
    # Optimization parameters
    optimization_params = {
        "num_iterations": 50,
        "learning_rate": 0.5,
        "samples_per_tx": 1_000_000,
        "max_depth": 13,
        "use_soft_min": True,
        "temperature": 0.2,
    }
    
    # Save configuration
    config = {
        "num_workers": NUM_WORKERS,
        "gpu_fraction": GPU_FRACTION,
        "optimizer_method": "gradient_descent",
        "scene_config": scene_config,
        "bounds": bounds,
        "optimization_params": optimization_params,
        "timestamp": timestamp,
    }
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("\nConfiguration saved")
    
    # Run parallel optimization
    try:
        results = parallel_optimizer.optimize(
            scene_config=scene_config,
            initial_positions=initial_positions,
            optimization_params=optimization_params,
            verbose=True,
        )
        
        # Save results
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            "best_worker_id": results["best_worker_id"],
            "total_time": results["total_time"],
            "aggregate_stats": results["aggregate_stats"],
            "best_result": {
                "worker_id": results["best_result"]["worker_id"],
                "best_position": results["best_result"]["best_position"].tolist(),
                "best_metric": float(results["best_result"]["best_metric"]),
                "time_elapsed": results["best_result"]["time_elapsed"],
            },
            "all_metrics": [
                float(r["best_metric"]) for r in results["all_results"]
            ],
            "all_positions": [
                r["best_position"].tolist() for r in results["all_results"]
            ],
        }
        
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {run_dir}")
        
        # Plot and save figure
        import matplotlib.pyplot as plt
        parallel_optimizer.plot_results(results)
        plt.savefig(os.path.join(run_dir, "results_plot.png"), dpi=150)
        print(f"Plot saved to: {os.path.join(run_dir, 'results_plot.png')}")
        
    except Exception as e:
        print(f"\nError during optimization: {e}")
        raise
    
    finally:
        # Always cleanup
        parallel_optimizer.shutdown()
    
    return results


if __name__ == "__main__":
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Run examples (uncomment the ones you want to run)
    
    # Example 1: Basic parallel optimization
    results_1 = example_basic_parallel_optimization()
    
    # Example 2: Parallel grid search
    # results_2 = example_grid_search_parallel()
    
    # Example 3: Hyperparameter search
    # results_3 = example_hyperparameter_search()
    
    # Example 4: Production workflow
    # results_4 = example_production_workflow()
    
    # Shutdown Ray
    ray.shutdown()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
