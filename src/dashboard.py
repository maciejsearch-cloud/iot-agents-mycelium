#!/usr/bin/env python3
"""
Streamlit Dashboard - Live monitoring systemu IoT Agents
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import time
import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from memory import MyceliumMemory
from tasks import TaskFactory
from universal_agent import UniversalAgent
from network_visualizer import NetworkVisualizer
from iq_calculator import AgentIQCalculator
from persistent_agent import get_persistent_agent
from auto_evolution import AutoEvolutionSystem


# Konfiguracja strony
st.set_page_config(
    page_title="IoT Agents Dashboard",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_mycelium_stats():
    """Ładuje statystyki z grzybni"""
    try:
        mycelium = MyceliumMemory("../data/mycelium_memory.json")
        return mycelium.get_stats(), mycelium.get_metadata()
    except Exception as e:
        return None, None


def train_agent_sync(task_name: str, hidden_dim: int, learning_rate: float, epochs: int, alpha: float):
    """Prawdziwy trening agenta z ciągłą pamięcią"""
    try:
        # Pobierz persistent agenta (ten sam między sesjami!)
        persistent_agent = get_persistent_agent(task_name, hidden_dim, learning_rate)
        
        # Kontynuuj trening (prawdziwe uczenie!)
        results = persistent_agent.continue_training(epochs=epochs)
        
        # Dodaj dodatkowe metryki
        learning_stats = persistent_agent.get_learning_stats()
        results.update(learning_stats)
        
        results['task_name'] = task_name
        results['agent'] = persistent_agent.agent
        results['persistent_agent'] = persistent_agent

        return results

    except Exception as e:
        # Obsługa błędów - zwróć informacje o błędzie
        return {
            'error': str(e),
            'task_name': task_name,
            'final_loss': float('inf'),
            'final_accuracy': 0.0,
            'training_time': 0.0,
            'updated_mycelium': False,
            'agent': None,
            'total_sessions': 0,
            'learning_speed': 0.0
        }


def calculate_agent_iq(agent) -> dict:
    """Oblicza IQ agenta i zwraca wyniki"""
    try:
        iq_calculator = AgentIQCalculator()
        iq_results = iq_calculator.calculate_iq(agent)
        return iq_results
    except Exception as e:
        return {
            'total_iq': 0,
            'breakdown': {'boolean_logic': 0, 'pattern_recognition': 0, 'math_foundation': 0, 'sequence_logic': 0, 'memory_retention': 0},
            'stage': {'name': 'foundation', 'description': 'Logika podstawowa', 'progress': 0.0},
            'next_stage_requirements': {'next_stage': 'patterns', 'needed_iq': 51},
            'evolution_progress': 0.0,
            'error': str(e)
        }


def main():
    # Header
    st.markdown('<h1 class="main-header">🍄 IoT Agents Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar - Kontrolki
    with st.sidebar:
        st.header("⚙️ Konfiguracja")

        # Wybór zadania
        tasks_info = TaskFactory.list_tasks()
        task_options = list(tasks_info.keys())
        selected_task = st.selectbox(
            "Wybierz zadanie",
            task_options,
            format_func=lambda x: f"{x.upper()} ({tasks_info[x]['difficulty']})"
        )

        st.info(f"**Opis:** {tasks_info[selected_task]['description']}")

        st.markdown("---")

        # Parametry treningu
        st.subheader("Parametry treningu")

        hidden_dim = st.slider("Neurony ukryte", 2, 16, 4, 1)
        learning_rate = st.slider("Learning rate", 0.1, 1.0, 0.5, 0.1)
        epochs = st.slider("Epoki", 1000, 20000, 5000, 1000)

        st.markdown("---")

        # Parametry Mycelium
        st.subheader("Mycelium Memory")

        alpha = st.slider(
            "Alpha (fusion)",
            0.0, 1.0, 0.7, 0.1,
            help="0.0 = tylko losowe wagi, 1.0 = tylko wagi z grzybni"
        )

        st.markdown("---")

        # Przyciski akcji
        train_button = st.button("🏃 Trenuj Agenta", type="primary", use_container_width=True)
        auto_evo_button = st.button("🤖 Auto-Evolution", type="secondary", use_container_width=True)
        reset_button = st.button("🔄 Reset Grzybni", type="secondary", use_container_width=True)

        if reset_button:
            mycelium = MyceliumMemory("../data/mycelium_memory.json")
            mycelium.reset()
            st.success("✅ Grzybnia zresetowana!")
            st.rerun()

    # Main content
    col1, col2, col3 = st.columns(3)

    # Statystyki Mycelium
    stats, metadata = load_mycelium_stats()

    if stats:
        with col1:
            st.metric(
                "🍄 Status Grzybni",
                "Pusta" if stats['is_empty'] else "Aktywna",
                delta=None
            )

        with col2:
            st.metric(
                "📉 Najlepszy Loss",
                f"{stats['best_loss']:.6f}" if not stats['is_empty'] else "N/A",
                delta=None
            )

        with col3:
            st.metric(
                "🔄 Aktualizacje",
                stats['total_updates'],
                delta=None
            )

    st.markdown("---")

    # Sekcja Auto-Evolution
    st.write(f"Debug: auto_evo_button = {auto_evo_button}")
    
    if auto_evo_button:
        st.write("Debug: Wchodzę do sekcji Auto-Evolution")
        st.markdown("## 🤖 Auto-Evolution System")
        
        # Debug info
        st.write("Debug: Przed przyciskiem Debug System")
        if st.button("🔍 Debug System", type="secondary"):
            st.write("Debug: Kliknięto Debug System")
            try:
                evolution_system = AutoEvolutionSystem()
                st.success("✅ System zainicjalizowany!")
                st.json({
                    'tasks': list(evolution_system.tasks.keys()),
                    'strategies': evolution_system.task_strategies,
                    'mastery_threshold': evolution_system.mastery_threshold,
                    'max_attempts': evolution_system.max_attempts_per_task
                })
            except Exception as e:
                st.error(f"❌ Błąd inicjalizacji: {e}")
        
        # Informacje o systemie
        st.info("""
        **🤖 Auto-Evolution** automatycznie nauczy agenta WSZYSTKICH wzorów:
        - AND, OR, NOT (proste)
        - XOR, NAND, NOR (średnie)
        - Inteligentna optymalizacja parametrów
        - Cel: 95% accuracy na wszystkich zadaniach
        """)
        
        # Potwierdzenie
        st.write("Debug: Przed przyciskiem Start Evolution")
        start_evolution = st.button("🚀 Start Auto-Evolution", type="primary")
        st.write(f"Debug: start_evolution = {start_evolution}")
        
        if start_evolution:
            st.write("Debug: Kliknięto Start Evolution")
            # Inicjalizuj system
            evolution_system = AutoEvolutionSystem()
            
            # Placeholder na status
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            results_placeholder = st.empty()
            
            def progress_callback(message):
                status_placeholder.markdown(f"### {message}")
            
            # Uruchom ewolucję
            try:
                with st.spinner("🤖 Auto-Evolution w toku... To może zająć 15-30 minut"):
                    status_placeholder.markdown("### 🚀 Inicjalizuję system ewolucji...")
                    evolution_results = evolution_system.run_full_evolution(progress_callback)
                    
                    # Pokaż wyniki
                    status_placeholder.markdown("### ✅ Auto-Evolution zakończone!")
                    
            except Exception as e:
                status_placeholder.markdown(f"### ❌ Błąd Auto-Evolution: {str(e)}")
                st.error(f"Wystąpił błąd: {e}")
                st.code(f"""
                Debug info:
                - System: {evolution_system}
                - Tasks: {list(evolution_system.tasks.keys()) if evolution_system else 'None'}
                - Error: {str(e)}
                """)
                return
                
            # Podsumowanie
                st.markdown("## 📊 Podsumowanie Ewolucji")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🎯 Opanowane zadania",
                        f"{len(evolution_results['tasks_mastered'])}/{len(evolution_results['tasks_mastered']) + len(evolution_results['tasks_failed'])}"
                    )
                
                with col2:
                    st.metric(
                        "⏱️ Całkowity czas",
                        f"{evolution_results['total_time']:.1f}s"
                    )
                
                with col3:
                    st.metric(
                        "🎓 Sesji treningu",
                        evolution_results['total_sessions']
                    )
                
                with col4:
                    success_rate = len(evolution_results['tasks_mastered']) / (len(evolution_results['tasks_mastered']) + len(evolution_results['tasks_failed'])) * 100
                    st.metric(
                        "📈 Sukces rate",
                        f"{success_rate:.1f}%"
                    )
                
                # Szczegółowe wyniki
                with st.expander("🔍 Szczegółowe wyniki zadań"):
                    for task_log in evolution_results['evolution_log']:
                        task_name = task_log['task_name'].upper()
                        success = task_log['success']
                        best_acc = task_log['best_accuracy']
                        attempts = len(task_log['attempts'])
                        
                        if success:
                            st.markdown(f"✅ **{task_name}**: {best_acc:.3f} ({attempts} prób)")
                        else:
                            st.markdown(f"❌ **{task_name}**: {best_acc:.3f} ({attempts} prób)")
                
                # Optymalne parametry
                st.markdown("## 🎯 Optymalne Parametry")
                optimization_summary = evolution_system.get_optimization_summary()
                
                for task_name, summary in optimization_summary.items():
                    with st.expander(f"📊 {task_name.upper()} - Optymalizacja"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Najlepsza dokładność", f"{summary['best_accuracy']:.3f}")
                            st.metric("Liczba prób", summary['attempts_needed'])
                        
                        with col2:
                            if summary['optimal_params']:
                                st.markdown("**Optymalne parametry:**")
                                st.code(f"""
Hidden Dim: {summary['optimal_params']['hidden_dim']}
Learning Rate: {summary['optimal_params']['learning_rate']}
Epochs: {summary['optimal_params']['epochs']}
""")

    st.markdown("---")

    # Sekcja treningu
    if train_button:
        # Pobierz obiekt task przed treningiem
        task = TaskFactory.get_task(selected_task)

        with st.spinner(f"🏃 Trenuję agenta na zadaniu {selected_task.upper()}..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Trening
            start_time = time.time()
            results = train_agent_sync(
                task_name=selected_task,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                epochs=epochs,
                alpha=alpha
            )
            end_time = time.time()

            progress_bar.progress(100)

            # Sprawdź czy wystąpił błąd
            if 'error' in results:
                status_text.text("❌ Wystąpił błąd podczas treningu")
                st.error(f"Błąd: {results['error']}")
                st.markdown("### 🔧 Możliwe rozwiązania:")
                st.markdown("- Spróbuj zresetować grzybnię (przycisk '🔄 Reset Grzybni')")
                st.markdown("- Zmniejsz liczbę neuronów ukrytych")
                st.markdown("- Sprawdź czy parametry są poprawne")
            else:
                status_text.text(f"✅ Trening zakończony w {end_time - start_time:.2f}s")

                # Wyniki
                st.markdown("## 📊 Wyniki Treningu")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Dokładność", f"{results['final_accuracy']:.3f}")

                with col2:
                    st.metric("Loss", f"{results['final_loss']:.6f}")

                with col3:
                    st.metric("Czas", f"{results['training_time']:.2f}s")

                with col4:
                    update_status = "✅ TAK" if results['updated_mycelium'] else "⚪ NIE"
                    st.metric("Aktualizacja Grzybni", update_status)

            # Komunikat o wyniku
                if results['final_accuracy'] >= 0.95:
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>✅ Sukces!</strong> Agent nauczył się zadania <strong>{selected_task.upper()}</strong>
                        z dokładnością {results['final_accuracy']:.1%}!
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                        <strong>⚠️ Uwaga!</strong> Agent osiągnął tylko {results['final_accuracy']:.1%} dokładności.
                        Spróbuj zwiększyć liczbę epok lub neuronów ukrytych.
                    </div>
                    """, unsafe_allow_html=True)

                # Wykresy i predykcje - tylko jeśli nie było błędu
                if 'agent' in results and results['agent'] is not None:
                    agent = results['agent']

                    col1, col2 = st.columns(2)

                    with col1:
                        # Wykres Loss
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            y=agent.loss_history,
                            mode='lines',
                            name='Loss',
                            line=dict(color='#ff6b6b', width=2)
                        ))
                        fig_loss.update_layout(
                            title="Krzywa Straty",
                            xaxis_title="Krok (x100)",
                            yaxis_title="Loss",
                            height=400
                        )
                        st.plotly_chart(fig_loss, width='stretch')

                    with col2:
                        # Wykres Accuracy
                        fig_acc = go.Figure()
                        fig_acc.add_trace(go.Scatter(
                            y=agent.accuracy_history,
                            mode='lines',
                            name='Accuracy',
                            line=dict(color='#51cf66', width=2)
                        ))
                        fig_acc.update_layout(
                            title="Dokładność",
                            xaxis_title="Krok (x100)",
                            yaxis_title="Accuracy",
                            yaxis=dict(range=[0, 1.1]),
                            height=400
                        )
                        st.plotly_chart(fig_acc, width='stretch')

                    # Predykcje
                    st.markdown("## 🎯 Predykcje")

                    eval_results = agent.evaluate(verbose=False)

                    # Tabela z wynikami
                    df = pd.DataFrame({
                        'Wejście': [str(x) for x in agent.X.tolist()],
                        'Oczekiwane': agent.y.flatten().tolist(),
                        'Predykcja (raw)': eval_results['raw_predictions'].flatten().tolist(),
                        'Predykcja (binary)': eval_results['predictions'].flatten().tolist(),
                        'Poprawne': (eval_results['predictions'].flatten() == agent.y.flatten()).tolist()
                    })

                    # Kolorowanie
                    def highlight_correct(row):
                        if row['Poprawne']:
                            return ['background-color: #d4edda'] * len(row)
                        else:
                            return ['background-color: #f8d7da'] * len(row)

                    st.dataframe(
                        df.style.apply(highlight_correct, axis=1).format({
                            'Predykcja (raw)': '{:.4f}'
                        }),
                        width='stretch',
                        hide_index=True
                    )
                    
                    # 📊 Learning Stats - Prawdziwe metryki uczenia!
                    st.markdown("## 📊 Learning Stats - Prawdziwe Uczenie")
                    
                    # Pobierz statystyki uczenia
                    if 'persistent_agent' in results:
                        learning_stats = results['persistent_agent'].get_learning_stats()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "🎓 Sesji Treningu",
                                learning_stats['total_sessions']
                            )
                        
                        with col2:
                            st.metric(
                                "⏱️ Całkowity czas",
                                f"{learning_stats['total_time']:.1f}s"
                            )
                        
                        with col3:
                            st.metric(
                                "📈 Prędkość uczenia",
                                f"{learning_stats['average_speed']:.4f}/s"
                            )
                        
                        with col4:
                            trend_emoji = {
                                'improving': '📈',
                                'declining': '📉', 
                                'stable': '➡️',
                                'insufficient_data': '❓'
                            }.get(learning_stats['improvement_trend'], '❓')
                            
                            st.metric(
                                f"Trend {trend_emoji}",
                                learning_stats['improvement_trend'].title()
                            )
                        
                        # Szczegółowe statystyki
                        with st.expander("🔍 Szczegółowe statystyki uczenia"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Postęp:**")
                                st.metric("Pierwsza dokładność", f"{learning_stats['first_accuracy']:.3f}")
                                st.metric("Najlepsza dokładność", f"{learning_stats['best_accuracy']:.3f}")
                                st.metric("Całkowita poprawa", f"{learning_stats['total_improvement']:+.3f}")
                                st.metric("Łącznie epok", learning_stats['total_epochs'])
                            
                            with col2:
                                st.markdown("**Efektywność:**")
                                if learning_stats['total_epochs'] > 0:
                                    accuracy_per_epoch = learning_stats['total_improvement'] / learning_stats['total_epochs']
                                    st.metric("Poprawa/epokę", f"{accuracy_per_epoch:.6f}")
                                
                                if learning_stats['total_time'] > 0:
                                    accuracy_per_second = learning_stats['total_improvement'] / learning_stats['total_time']
                                    st.metric("Poprawa/sekundę", f"{accuracy_per_second:.6f}")
                                
                                if learning_stats['total_sessions'] > 0:
                                    epochs_per_session = learning_stats['total_epochs'] / learning_stats['total_sessions']
                                    st.metric("Epok/sesję", f"{epochs_per_session:.0f}")
                        
                        # Historia sesji
                        if learning_stats['total_sessions'] > 1:
                            st.markdown("### 📈 Historia Sesji")
                            
                            # Wykres postępu w sesjach
                            session_accuracies = [s['final_accuracy'] for s in results['persistent_agent'].training_sessions]
                            session_numbers = list(range(1, len(session_accuracies) + 1))
                            
                            fig_sessions = go.Figure()
                            fig_sessions.add_trace(go.Scatter(
                                x=session_numbers,
                                y=session_accuracies,
                                mode='lines+markers',
                                name='Dokładność w sesjach',
                                line=dict(color='#ff6b6b', width=2),
                                marker=dict(size=8)
                            ))
                            
                            fig_sessions.update_layout(
                                title="Postęp w sesjach treningowych",
                                xaxis_title="Numer sesji",
                                yaxis_title="Dokładność",
                                height=300
                            )
                            
                            st.plotly_chart(fig_sessions, width='stretch')

                    # Wizualizacja sieci neuronowej
                    st.markdown("## 🧠 Wizualizacja Sieci Neuronowej")

                    # Stwórz wizualizator
                    visualizer = NetworkVisualizer(
                        input_dim=task.input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=task.output_dim
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 🎯 Aktualna Architektura")
                        # Ostatnie aktywacje
                        if agent.activation_history:
                            last_activations = agent.activation_history[-1]
                            network_fig = visualizer.create_network_plot(
                                weights=agent.get_weights(),
                                activations=last_activations,
                                title=f"Sieć: {task.name.upper()} ({task.input_dim}→{hidden_dim}→{task.output_dim})"
                            )
                            st.plotly_chart(network_fig, width='stretch')

                    with col2:
                        st.markdown("### 📈 Ewolucja Wag")
                        if agent.weight_history and len(agent.weight_history) > 1:
                            evolution_fig = visualizer.create_weight_evolution_plot(
                                agent.weight_history,
                                title="Jak wagi zmieniają się w czasie"
                            )
                            st.plotly_chart(evolution_fig, width='stretch')

                    # Heatmap aktywacji
                    if agent.activation_history:
                        st.markdown("### 🔥 Heatmap Aktywacji")
                        last_activations = agent.activation_history[-1]
                        heatmap_fig = visualizer.create_activation_heatmap(
                            last_activations,
                            title="Aktywacje neuronów dla ostatniego batcha"
                        )
                        st.plotly_chart(heatmap_fig, width='stretch')

                    # Informacje o wagach
                    st.markdown("### 📊 Statystyki Wag")
                    weights = agent.get_weights()
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "W1 średnia",
                            f"{np.mean(np.abs(weights['W1'])):.4f}"
                        )

                    with col2:
                        st.metric(
                            "W1 odchylenie",
                            f"{np.std(weights['W1']):.4f}"
                        )

                    with col3:
                        st.metric(
                            "W2 średnia",
                            f"{np.mean(np.abs(weights['W2'])):.4f}"
                        )

                    with col4:
                        st.metric(
                            "W2 odchylenie",
                            f"{np.std(weights['W2']):.4f}"
                        )
                    
                    # 🧠 IQ Agent Section
                    st.markdown("## 🧠 IQ Agent")
                    
                    # Oblicz IQ
                    iq_results = calculate_agent_iq(agent)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "🧠 Aktualne IQ", 
                            f"{iq_results['total_iq']:.0f}",
                            delta=None
                        )
                    
                    with col2:
                        stage_name = iq_results['stage']['name'].upper()
                        st.metric(
                            "🎯 Etap Ewolucji", 
                            stage_name,
                            delta=None
                        )
                    
                    with col3:
                        next_iq = iq_results['next_stage_requirements'].get('needed_iq', 0)
                        st.metric(
                            "📈 IQ do następnego etapu", 
                            f"{next_iq:.0f}",
                            delta=None
                        )
                    
                    # Progress bar ewolucji
                    st.markdown("### 📊 Postęp Ewolucji")
                    
                    # Ogólny postęp
                    overall_progress = iq_results['evolution_progress']
                    st.progress(overall_progress, f"Ogólny postęp: {overall_progress*100:.1f}%")
                    
                    # Postęp w aktualnym etapie
                    stage_progress = iq_results['stage']['progress']
                    st.progress(stage_progress, f"Postęp w etapie: {stage_progress*100:.1f}%")
                    
                    # Szczegółowe wyniki IQ
                    with st.expander("🔍 Szczegółowe wyniki IQ"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Testy IQ:**")
                            for test_name, score in iq_results['breakdown'].items():
                                test_display = test_name.replace('_', ' ').title()
                                max_score = 30 if test_name != 'memory_retention' else 10
                                st.metric(test_display, f"{score:.0f}/{max_score}")
                        
                        with col2:
                            st.markdown("**Wymagania do następnego etapu:**")
                            next_req = iq_results['next_stage_requirements']
                            if next_req['next_stage'] != 'max':
                                st.info(f"""
                                **Następny etap:** {next_req['next_stage'].upper()}
                                **Opis:** {next_req['description']}
                                **Potrzebne IQ:** {next_req['target_iq']}
                                **Brakuje:** {next_req['needed_iq']} punktów
                                """)
                            else:
                                st.success("🏆 Osiągnięto maksymalny etap ewolucji!")

    # Informacje o grzybni
    st.markdown("---")
    st.markdown("## 🍄 Historia Mycelium")

    if stats and not stats['is_empty']:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Metadane")
            if metadata:
                st.json({
                    'Całkowite aktualizacje': metadata.get('total_updates', 0),
                    'Ostatnia aktualizacja': metadata.get('last_updated', 'N/A'),
                    'Utworzono': metadata.get('created_at', 'N/A'),
                    'Najlepszy agent': metadata.get('agent_id', 'N/A'),
                    'Zadanie': metadata.get('task_name', 'N/A'),
                    'Dokładność': metadata.get('accuracy', 'N/A')
                })

        with col2:
            st.markdown("### Statystyki")
            st.metric("Najlepszy Loss", f"{stats['best_loss']:.6f}")
            st.metric("Całkowite aktualizacje", stats['total_updates'])
    else:
        st.info("🍄 Grzybnia jest pusta. Wytrenuj pierwszego agenta!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🤖 IoT Agents Project | Mycelium Memory Architecture | Python + NumPy</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
