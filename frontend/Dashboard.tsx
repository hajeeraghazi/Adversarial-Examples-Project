"use client";

import axios from "axios";
import {
  AlertTriangle,
  Database,
  Play,
  Shield,
  Sparkles,
  Target,
} from "lucide-react";
import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

interface ExperimentResult {
  cleanAccuracy: number;
  attackSuccessRate: number;
  robustAccuracy: number;
  perturbationMagnitude: number;
}

export default function AdversarialUI() {
  const [activeTab, setActiveTab] = useState<
    "overview" | "attacks" | "defenses" | "experiment"
  >("experiment");

  const [attackType, setAttackType] = useState<"fgsm" | "pgd">("fgsm");
  const [epsilon, setEpsilon] = useState<number>(0.1);

  const [defenseType, setDefenseType] = useState<
    "none" | "adversarial" | "input_transform" | "ensemble"
  >("none");

  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<ExperimentResult | null>(null);

  const [history, setHistory] = useState<
    { epsilon: number; clean: number; robust: number; attackSuccess: number }[]
  >([]);

  // Static fallback charts
  const attackChart = [
    { eps: "0.00", fgsm: 0, pgd: 0 },
    { eps: "0.05", fgsm: 34, pgd: 42 },
    { eps: "0.10", fgsm: 68, pgd: 78 },
    { eps: "0.15", fgsm: 83, pgd: 91 },
    { eps: "0.20", fgsm: 89, pgd: 96 },
    { eps: "0.30", fgsm: 94, pgd: 98 },
  ];

  const defenseChart = [
    { method: "None", clean: 92, fgsm: 23, pgd: 12 },
    { method: "Adversarial", clean: 87, fgsm: 68, pgd: 61 },
    { method: "Transform", clean: 89, fgsm: 52, pgd: 45 },
    { method: "Ensemble", clean: 88, fgsm: 71, pgd: 64 },
  ];

  async function runExperiment() {
    setIsLoading(true);
    setResults(null);

    try {
      const payload = {
        attack_type: attackType,
        epsilon,
        defense_type: defenseType,
        num_samples: 300,
      };

      const res = await axios.post(`${API_URL}/api/run_experiment`, payload, {
        headers: { "Content-Type": "application/json" },
      });

      const r: ExperimentResult = {
        cleanAccuracy: res.data.clean_accuracy,
        attackSuccessRate: res.data.attack_success_rate,
        robustAccuracy: res.data.robust_accuracy,
        perturbationMagnitude: epsilon,
      };

      setResults(r);
      setHistory((p) => [
        ...p,
        {
          epsilon,
          clean: r.cleanAccuracy,
          robust: r.robustAccuracy,
          attackSuccess: r.attackSuccessRate,
        },
      ]);
    } catch (err) {
      console.error("Backend error:", err);
      alert("Backend error — check console & CORS.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-purple-950 to-black text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* HEADER */}
        <header className="text-center mb-12">
          <h1 className="text-5xl font-extrabold neon-text tracking-tight">
            Adversarial ML Dashboard
          </h1>
          <p className="text-gray-300 mt-2">
            FGSM • PGD • Defense evaluation on CIFAR-10
          </p>
        </header>

        {/* NEON NAVBAR */}
        <nav className="relative flex justify-center mb-10 mt-4">
          <div className="flex gap-3 p-2 glass-card rounded-2xl shadow-card border border-white/10 backdrop-blur-xl">
            {[
              { id: "overview", label: "Overview", icon: AlertTriangle },
              { id: "attacks", label: "Attacks", icon: Target },
              { id: "defenses", label: "Defenses", icon: Shield },
              { id: "experiment", label: "Run Experiment", icon: Play },
            ].map((tab) => {
              const active = activeTab === tab.id;

              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`
                    relative px-5 py-2 rounded-xl font-semibold flex items-center gap-2 
                    transition-all duration-300
                    ${active ? "text-white" : "text-gray-300 hover:text-white"}
                  `}
                >
                  {/* Neon active border */}
                  {active && (
                    <span className="absolute inset-0 rounded-xl border border-purple-500/40 shadow-[0_0_18px_rgba(168,85,247,0.6)] animate-pulse"></span>
                  )}

                  {/* Inner container */}
                  <span
                    className={`
                      relative z-10 flex items-center gap-2 px-2 py-1
                      ${
                        active
                          ? "bg-purple-600/40 backdrop-blur-md rounded-xl"
                          : ""
                      }
                    `}
                  >
                    <tab.icon size={17} />
                    {tab.label}
                  </span>

                  {/* Gradient underline */}
                  {active && (
                    <div
                      className="absolute -bottom-1 left-1/2 w-2/3 h-[3px]
                      bg-gradient-to-r from-neon-purple via-neon-pink to-neon-blue
                      rounded-full -translate-x-1/2 animate-gradientMove"
                    />
                  )}
                </button>
              );
            })}
          </div>
        </nav>

        {/* CONTENT WRAPPER */}
        <main className="glass-card p-8">
          {/* ----------------------------- OVERVIEW ----------------------------- */}
          {activeTab === "overview" && (
            <section className="space-y-6">
              <h2 className="text-3xl font-bold neon-text flex items-center gap-3">
                <AlertTriangle className="text-yellow-400" />
                What are adversarial examples?
              </h2>

              <p className="text-gray-300 leading-relaxed">
                Adversarial examples are tiny, carefully crafted perturbations
                that cause neural networks to misclassify inputs. This dashboard
                demonstrates attacks, defenses, and robustness evaluation.
              </p>

              <div className="grid md:grid-cols-2 gap-6 mt-6">
                <div className="glass-card p-5 card-hover">
                  <h3 className="font-semibold text-lg text-purple-200">
                    Key Concepts
                  </h3>
                  <ul className="mt-3 text-gray-300 space-y-1">
                    <li>• FGSM — single-step gradient attack</li>
                    <li>• PGD — iterative stronger attack</li>
                    <li>• Adversarial Training — improve robustness</li>
                    <li>• Input Transformations — denoising defense</li>
                    <li>• Ensemble — logits voting</li>
                  </ul>
                </div>

                <div className="glass-card p-5 card-hover">
                  <h3 className="font-semibold text-lg text-pink-200">
                    What You Can Do
                  </h3>
                  <ol className="mt-3 text-gray-300 list-decimal list-inside space-y-1">
                    <li>Run experiments (attack, ε, defense)</li>
                    <li>Visualize robustness trends</li>
                    <li>Use plots in your IEEE/VTU report</li>
                  </ol>
                </div>
              </div>
            </section>
          )}

          {/* ----------------------------- ATTACKS ----------------------------- */}
          {activeTab === "attacks" && (
            <section>
              <h2 className="text-3xl font-bold neon-text mb-6">
                Attack Effectiveness
              </h2>

              <div className="glass-card p-5">
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={attackChart}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2b2b2b" />
                    <XAxis dataKey="eps" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip
                      contentStyle={{ background: "#0b1220", border: "none" }}
                    />
                    <Legend />
                    <Line dataKey="fgsm" stroke="#ff44cc" strokeWidth={2} />
                    <Line dataKey="pgd" stroke="#4fd6ff" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}

          {/* ----------------------------- DEFENSES ----------------------------- */}
          {activeTab === "defenses" && (
            <section>
              <h2 className="text-3xl font-bold neon-text mb-6">
                Defense Comparison
              </h2>

              <div className="glass-card p-5">
                <ResponsiveContainer width="100%" height={360}>
                  <BarChart data={defenseChart}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2b2b2b" />
                    <XAxis dataKey="method" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip
                      contentStyle={{ background: "#0b1220", border: "none" }}
                    />
                    <Legend />
                    <Bar dataKey="clean" fill="#10b981" />
                    <Bar dataKey="fgsm" fill="#ef4444" />
                    <Bar dataKey="pgd" fill="#f97316" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}

          {/* ----------------------------- EXPERIMENT ----------------------------- */}
          {activeTab === "experiment" && (
            <section className="space-y-8">
              {/* Form */}
              <div className="grid md:grid-cols-3 gap-6">
                {/* Attack Type */}
                <div className="glass-card p-4">
                  <label className="text-sm text-gray-300">Attack Type</label>
                  <select
                    value={attackType}
                    onChange={(e) => setAttackType(e.target.value as any)}
                    className="mt-2 w-full p-2 rounded bg-slate-900 border border-slate-700"
                  >
                    <option value="fgsm">FGSM</option>
                    <option value="pgd">PGD</option>
                  </select>
                </div>

                {/* Epsilon Slider */}
                <div className="glass-card p-4">
                  <label className="text-sm text-gray-300">
                    Epsilon (ε): {epsilon.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={0.3}
                    step={0.01}
                    value={epsilon}
                    onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                    className="mt-3 w-full accent-purple-500"
                  />
                </div>

                {/* Defense */}
                <div className="glass-card p-4">
                  <label className="text-sm text-gray-300">Defense</label>
                  <select
                    value={defenseType}
                    onChange={(e) => setDefenseType(e.target.value as any)}
                    className="mt-2 w-full p-2 rounded bg-slate-900 border border-slate-700"
                  >
                    <option value="none">No Defense</option>
                    <option value="adversarial">Adversarial Training</option>
                    <option value="input_transform">
                      Input Transformation
                    </option>
                    <option value="ensemble">Ensemble</option>
                  </select>
                </div>
              </div>

              {/* Buttons */}
              <div className="flex gap-4 items-center">
                <button
                  onClick={runExperiment}
                  disabled={isLoading}
                  className="gradient-btn flex items-center gap-3 px-6 py-3 rounded-xl font-semibold"
                >
                  {isLoading ? (
                    <span className="inline-flex h-5 w-5 rounded-full border-t-2 border-white animate-spin" />
                  ) : (
                    <Play size={16} />
                  )}
                  {isLoading ? "Running..." : "Run Experiment"}
                </button>

                <button
                  onClick={() => {
                    setResults(null);
                    setHistory([]);
                  }}
                  className="px-5 py-3 rounded-xl bg-slate-800/70 hover:bg-slate-800/90"
                >
                  Reset
                </button>

                <div className="ml-auto flex items-center gap-2 text-sm text-gray-300">
                  <Database size={16} />
                  <span>API: {API_URL.replace(/^https?:\/\//, "")}</span>
                </div>
              </div>

              {/* Results */}
              {results && (
                <div className="grid md:grid-cols-4 gap-4 mt-6">
                  {[
                    {
                      label: "Clean Accuracy",
                      value: results.cleanAccuracy,
                      color: "text-green-400",
                    },
                    {
                      label: "Attack Success",
                      value: results.attackSuccessRate,
                      color: "text-red-400",
                    },
                    {
                      label: "Robust Accuracy",
                      value: results.robustAccuracy,
                      color: "text-blue-400",
                    },
                    {
                      label: "Epsilon",
                      value: results.perturbationMagnitude,
                      color: "text-purple-400",
                    },
                  ].map((r, i) => (
                    <div key={i} className="glass-card p-5 card-hover">
                      <div className={`text-3xl font-bold ${r.color}`}>
                        {r.value.toFixed(2)}
                      </div>
                      <div className="text-gray-300 mt-1">{r.label}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Dynamic Charts */}
              {history.length > 0 && (
                <>
                  <div className="glass-card p-5 mt-6">
                    <h4 className="font-semibold neon-text mb-3 flex items-center gap-2">
                      <Sparkles /> Attack Success vs Epsilon
                    </h4>
                    <ResponsiveContainer width="100%" height={240}>
                      <LineChart data={history}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                        <XAxis dataKey="epsilon" stroke="#9ca3af" />
                        <YAxis stroke="#9ca3af" />
                        <Tooltip />
                        <Legend />
                        <Line
                          dataKey="attackSuccess"
                          stroke="#f97316"
                          strokeWidth={2}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="glass-card p-5 mt-6">
                    <h4 className="font-semibold neon-text mb-3 flex items-center gap-2">
                      <Sparkles /> Clean vs Robust Accuracy
                    </h4>
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={history}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                        <XAxis dataKey="epsilon" stroke="#9ca3af" />
                        <YAxis stroke="#9ca3af" />
                        <Tooltip />
                        <Legend />
                        <Line
                          dataKey="clean"
                          stroke="#10b981"
                          strokeWidth={2}
                        />
                        <Line
                          dataKey="robust"
                          stroke="#3b82f6"
                          strokeWidth={2}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
