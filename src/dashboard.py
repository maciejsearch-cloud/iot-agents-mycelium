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
    """Synchroniczny trening agenta (dla Streamlit)"""
    try:
        # Pobierz zadanie
        task = TaskFactory.get_task(task_name)

        # Inicjalizuj Mycelium
        mycelium = MyceliumMemory("../data/mycelium_memory.json")

        # Pobierz wagi z grzybni
        shapes = {
            'W1': (task.input_dim, hidden_dim),
            'b1': (1, hidden_dim),
            'W2': (hidden_dim, task.output_dim),
            'b2': (1, task.output_dim)
        }

        initial_weights = mycelium.get_fusion_weights(shapes, alpha=alpha)

        # Stwórz agenta
        agent = UniversalAgent(
            task=task,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            initial_weights=initial_weights
        )

        # Trening
        results = agent.train(epochs=epochs, verbose=False)

        # Próba aktualizacji grzybni
        weights = agent.get_weights()
        updated = mycelium.update_memory(
            weights=weights,
            loss=results['final_loss'],
            metadata={
                'task_name': task_name,
                'accuracy': results['final_accuracy'],
                'training_time': results['training_time'],
                'hidden_dim': hidden_dim,
                'learning_rate': learning_rate
            }
        )

        results['updated_mycelium'] = updated
        results['agent'] = agent

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
            'agent': None
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
