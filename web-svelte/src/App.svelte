<script>
  import { onMount } from 'svelte';
  import { Chess } from 'chess.js';
  import { Chessboard, FEN, INPUT_EVENT_TYPE } from 'cm-chessboard';
  import "cm-chessboard/assets/chessboard.css";

  let boardElement;
  let chessboard;
  let game = new Chess();
  let status = "White to move";
  let evalScore = "0.0";
  let loading = false;

  onMount(async () => {
    chessboard = await new Chessboard(boardElement, {
      position: FEN.start,
      assetsUrl: "https://unpkg.com/cm-chessboard@8.5.0/assets/",
      style: {
        cssClass: "default",
      },
      responsive: true
    });

    chessboard.enableMoveInput(inputHandler);
  });

  async function inputHandler(event) {
    if (loading) return false;

    if (event.type === INPUT_EVENT_TYPE.moveInputStarted) {
      if (game.isGameOver()) return false;
      // Only picking up own pieces
      const piece = game.get(event.square);
      if (!piece || piece.color !== game.turn()) return false;
      return true;
    }

    if (event.type === INPUT_EVENT_TYPE.validateMoveInput) {
      const move = {
        from: event.squareFrom,
        to: event.squareTo,
        promotion: 'q' // Always promote to queen for simplicity
      };

      try {
        const result = game.move(move);
        if (result) {
          // Valid move
          updateStatus();
          makeBotMove(); // Trigger bot
          return true;
        }
      } catch (e) {
        return false;
      }
      return false;
    }
  }

  function updateStatus() {
    if (game.isCheckmate()) {
      status = "Game over: Checkmate!";
    } else if (game.isDraw()) {
      status = "Game over: Draw!";
    } else {
      status = (game.turn() === 'w' ? "White" : "Black") + " to move";
      if (game.isCheck()) status += " (Check)";
    }
    // Update board position cleanly
    chessboard.setPosition(game.fen());
  }

  async function makeBotMove() {
    if (game.isGameOver()) return;

    loading = true;
    status = "Bot thinking...";

    try {
      const response = await fetch("http://localhost:8000/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: game.fen() })
      });

      const data = await response.json();

      game.move(data.uci);
      evalScore = data.evaluation.toFixed(2);
      chessboard.setPosition(game.fen(), true); // Animated
      updateStatus();
    } catch (e) {
      status = "Error: " + e.message;
      console.error(e);
    } finally {
      loading = false;
    }
  }

  function newGame() {
    game.reset();
    chessboard.setPosition("start");
    status = "White to move";
    evalScore = "0.0";
    loading = false;
  }
</script>

<main>
  <div class="container">
    <h1>Chess RL Bot</h1>
    <div class="info">
      <span>{status}</span>
      <span>Eval: <span class="eval">{evalScore}</span></span>
    </div>

    <div class="board-wrapper" bind:this={boardElement}></div>

    <div class="controls">
      <button on:click={newGame} disabled={loading}>New Game</button>
      <button on:click={() => chessboard.setOrientation(chessboard.getOrientation() === 'w' ? 'b' : 'w')}>Flip Board</button>
    </div>
  </div>
</main>

<style>
  :global(body) {
    background-color: #222;
    color: #eee;
    font-family: 'Segoe UI', sans-serif;
    margin: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
  }

  .container {
    width: 100%;
    max-width: 600px;
    padding: 20px;
    text-align: center;
  }

  h1 { color: #4CAF50; }

  .board-wrapper {
    width: 100%;
    max-width: 500px;
    aspect-ratio: 1;
    margin: 20px auto;
  }

  .info {
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
    font-size: 1.2rem;
  }

  .eval { color: gold; font-weight: bold; }

  button {
    background: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 1rem;
    cursor: pointer;
    border-radius: 4px;
    margin: 0 10px;
  }

  button:disabled { background: #555; }
  button:hover:not(:disabled) { background: #45a049; }
</style>
